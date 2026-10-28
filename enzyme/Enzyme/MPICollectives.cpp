//===- MPICollectives.cpp - Differentiation of irregular collectives
//--------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "AdjointGenerator.h"
#include <functional>

using namespace llvm;
namespace {
// Metadata is discrete, but its values must survive mutations after the call.
// The fifth array holds compact reverse-receive offsets. Using original send
// displacements there would illegally overlap receives when primal sends alias.
struct AlltoallvBuilder {
  Module &M;
  Function *F;
  IRBuilder<> B;
  IntegerType *I32, *Index;
  AlltoallvBuilder(Module &M, Function *F)
      : M(M), F(F), B(BasicBlock::Create(M.getContext(), "entry", F)),
        I32(Type::getInt32Ty(M.getContext())),
        Index(M.getDataLayout().getIntPtrType(M.getContext())) {}
  Value *idx(Value *V) { return B.CreateSExtOrTrunc(V, Index); }
  Value *ptr(Type *T, Value *V) { return B.CreatePointerCast(V, getUnqual(T)); }
  Value *at(Type *T, Value *P, Value *I) {
    return B.CreateInBoundsGEP(T, ptr(T, P), idx(I));
  }
  void loop(Value *N, const Twine &Name,
            const std::function<void(Value *)> &Body) {
    auto Pre = B.GetInsertBlock();
    auto Head = BasicBlock::Create(M.getContext(), Name, F);
    auto End = BasicBlock::Create(M.getContext(), Name + ".end", F);
    B.CreateCondBr(B.CreateICmpSGT(N, ConstantInt::get(N->getType(), 0)), Head,
                   End);
    B.SetInsertPoint(Head);
    auto I = B.CreatePHI(N->getType(), 2);
    I->addIncoming(ConstantInt::get(N->getType(), 0), Pre);
    Body(I);
    auto Latch = B.GetInsertBlock();
    auto Next = B.CreateAdd(I, ConstantInt::get(N->getType(), 1));
    B.CreateCondBr(B.CreateICmpULT(Next, N), Head, End);
    I->addIncoming(Next, Latch);
    B.SetInsertPoint(End);
  }
  // Fail before accessing unsupported metadata, rather than emitting a wrong AD
  // program. This first implementation is the intracommunicator C interface.
  void require(Value *OK, const Twine &Reason) {
    auto Good = BasicBlock::Create(M.getContext(), "supported", F);
    auto Bad = BasicBlock::Create(M.getContext(), Reason, F);
    B.CreateCondBr(OK, Good, Bad);
    B.SetInsertPoint(Bad);
    B.CreateCall(getIntrinsicDeclaration(&M, Intrinsic::trap));
    B.CreateUnreachable();
    B.SetInsertPoint(Good);
  }
  Value *alloc(Type *T, Value *N) {
    auto Count =
        B.CreateSelect(B.CreateICmpSGT(N, ConstantInt::get(N->getType(), 0)), N,
                       ConstantInt::get(N->getType(), 1));
    return ptr(T, CreateAllocation(B, T, idx(Count)));
  }
};

Type *mpiRealType(Value *V) {
  auto C = dyn_cast<Constant>(V);
  if (!C)
    return nullptr;
  C = cast<Constant>(C->stripPointerCasts());
  if (auto G = dyn_cast<GlobalVariable>(C)) {
    if (G->getName() == "ompi_mpi_double")
      return Type::getDoubleTy(V->getContext());
    if (G->getName() == "ompi_mpi_float")
      return Type::getFloatTy(V->getContext());
  }
  if (auto I = dyn_cast<ConstantInt>(C)) {
    if (I->getValue() == 1275070475)
      return Type::getDoubleTy(V->getContext());
    if (I->getValue() == 1275069450)
      return Type::getFloatTy(V->getContext());
  }
  return nullptr;
}

StructType *alltoallvTape(Function *Call) {
  auto &C = Call->getContext();
  return StructType::get(C,
                         {getUnqual(Type::getInt32Ty(C)), Type::getInt32Ty(C),
                          Call->getFunctionType()->getParamType(8)});
}

Function *alltoallvSnapshot(Function *Call) {
  auto &M = *Call->getParent();
  auto Name = "__enzyme_alltoallv_snapshot_" + Call->getName().str();
  if (auto F = M.getFunction(Name))
    return F;
  auto T = alltoallvTape(Call);
  auto IT = Type::getInt32Ty(M.getContext());
  auto IP = getUnqual(IT);
  auto Comm = Call->getFunctionType()->getParamType(8);
  auto F = Function::Create(FunctionType::get(T, {IP, IP, IP, IP, Comm}, false),
                            GlobalValue::InternalLinkage, Name, M);
  AlltoallvBuilder G(M, F);
  auto &B = G.B;
  auto NPtr = B.CreateAlloca(IT);
  auto Inter = B.CreateAlloca(IT);
  auto QueryTy = FunctionType::get(IT, {Comm, IP}, false);
  B.CreateCall(
      M.getOrInsertFunction(
          getRenamedPerCallingConv(Call->getName(), "MPI_Comm_test_inter"),
          QueryTy),
      {F->getArg(4), Inter});
  G.require(B.CreateICmpEQ(B.CreateLoad(IT, Inter), B.getInt32(0)),
            "unsupported.intercommunicator");
  B.CreateCall(
      M.getOrInsertFunction(
          getRenamedPerCallingConv(Call->getName(), "MPI_Comm_size"), QueryTy),
      {F->getArg(4), NPtr});
  auto N = B.CreateLoad(IT, NPtr);
  auto Meta = G.alloc(IT, B.CreateMul(G.idx(N), ConstantInt::get(G.Index, 5)));
  for (unsigned A = 0; A < 4; ++A)
    G.loop(N, "save.metadata", [&](Value *I) {
      auto Offset = B.CreateAdd(
          G.idx(I), B.CreateMul(G.idx(N), ConstantInt::get(G.Index, A)));
      B.CreateStore(B.CreateLoad(IT, G.at(IT, F->getArg(A), I)),
                    G.at(IT, Meta, Offset));
    });
  Value *Tape = UndefValue::get(T);
  Tape = B.CreateInsertValue(Tape, Meta, 0);
  Tape = B.CreateInsertValue(Tape, N, 1);
  Tape = B.CreateInsertValue(Tape, F->getArg(4), 2);
  B.CreateRet(Tape);
  return F;
}

Function *alltoallvReverse(Function *Call, Type *FP, Value *Datatype,
                           bool FreeTape) {
  auto &M = *Call->getParent();
  auto Name = "__enzyme_alltoallv_reverse_" + Call->getName().str() +
              (FP->isDoubleTy() ? "_f64" : "_f32") + (FreeTape ? "_free" : "");
  if (auto F = M.getFunction(Name))
    return F;
  auto T = alltoallvTape(Call);
  auto F = Function::Create(FunctionType::get(Type::getVoidTy(M.getContext()),
                                              {T, getUnqual(FP), getUnqual(FP)},
                                              false),
                            GlobalValue::InternalLinkage, Name, M);
  AlltoallvBuilder G(M, F);
  auto &B = G.B;
  auto Tape = F->getArg(0);
  auto Meta = B.CreateExtractValue(Tape, 0);
  auto N = B.CreateExtractValue(Tape, 1);
  auto Comm = B.CreateExtractValue(Tape, 2);
  auto Array = [&](unsigned A) {
    return G.at(G.I32, Meta,
                B.CreateMul(G.idx(N), ConstantInt::get(G.Index, A)));
  };
  auto SC = Array(0), SD = Array(1), RC = Array(2), RD = Array(3),
       Packed = Array(4);
  auto Total = B.CreateAlloca(G.Index);
  B.CreateStore(ConstantInt::get(G.Index, 0), Total);
  G.loop(N, "compact.offsets", [&](Value *I) {
    auto Off = B.CreateLoad(G.Index, Total);
    auto Count = B.CreateLoad(G.I32, G.at(G.I32, SC, I));
    G.require(
        B.CreateAnd(B.CreateICmpSGE(Count, B.getInt32(0)),
                    B.CreateICmpSLE(
                        Off, B.CreateSub(ConstantInt::get(G.Index, INT32_MAX),
                                         G.idx(Count)))),
        "unsupported.packed.count.overflow");
    auto New = B.CreateAdd(Off, G.idx(Count));
    B.CreateStore(B.CreateIntCast(Off, G.I32, false), G.at(G.I32, Packed, I));
    B.CreateStore(New, Total);
  });
  auto Temp = G.alloc(FP, B.CreateLoad(G.Index, Total));
  SmallVector<Value *, 9> Args = {
      F->getArg(2), RC, RD, Datatype, Temp, SC, Packed, Datatype, Comm};
  auto FT = Call->getFunctionType();
  for (unsigned I = 0; I < Args.size(); ++I)
    if (Args[I]->getType() != FT->getParamType(I))
      Args[I] = B.CreatePointerCast(Args[I], FT->getParamType(I));
  B.CreateCall(FT, Call, Args);
  // Accumulate peer by peer: a primal element may have been sent repeatedly.
  G.loop(N, "accumulate.peer", [&](Value *I) {
    auto Count = B.CreateLoad(G.I32, G.at(G.I32, SC, I));
    auto Dst = B.CreateLoad(G.I32, G.at(G.I32, SD, I));
    auto Src = B.CreateLoad(G.I32, G.at(G.I32, Packed, I));
    G.loop(Count, "accumulate.element", [&](Value *J) {
      auto D = G.at(FP, F->getArg(1), B.CreateAdd(G.idx(Dst), G.idx(J)));
      auto S = G.at(FP, Temp, B.CreateAdd(G.idx(Src), G.idx(J)));
      B.CreateStore(B.CreateFAdd(B.CreateLoad(FP, D), B.CreateLoad(FP, S)), D);
    });
  });
  // Only the overwritten receive segments lose their pre-call adjoints.
  G.loop(N, "clear.peer", [&](Value *I) {
    auto Count = B.CreateLoad(G.I32, G.at(G.I32, RC, I));
    auto Off = B.CreateLoad(G.I32, G.at(G.I32, RD, I));
    G.loop(Count, "clear.element", [&](Value *J) {
      B.CreateStore(ConstantFP::get(FP, 0),
                    G.at(FP, F->getArg(2), B.CreateAdd(G.idx(Off), G.idx(J))));
    });
  });
  CreateDealloc(B, Temp);
  if (FreeTape)
    CreateDealloc(B, Meta);
  B.CreateRetVoid();
  return F;
}
} // namespace

bool AdjointGenerator::handleMPIAlltoallv(CallInst &call, Function *called,
                                          StringRef Name) {
  if (canonicalizeMPIName(Name) != "MPI_Alltoallv")
    return false;
  auto New = cast<CallInst>(gutils->getNewFromOriginal(&call));
  IRBuilder<> B(New);
  if (gutils->isConstantInstruction(&call))
    return true;
  bool ValidABI = called && call.arg_size() == 9 && !called->isVarArg() &&
                  called->arg_size() == 9 &&
                  call.getFunctionType() == called->getFunctionType() &&
                  call.getCallingConv() == CallingConv::C &&
                  called->getCallingConv() == CallingConv::C;
  if (ValidABI) {
    for (unsigned I : {0u, 1u, 2u, 4u, 5u, 6u}) {
      auto P = dyn_cast<PointerType>(call.getArgOperand(I)->getType());
      ValidABI &= P && P->getAddressSpace() == 0;
    }
  }
  auto FP = ValidABI ? mpiRealType(call.getArgOperand(3)) : nullptr;
  if (!called || (Name != "MPI_Alltoallv" && Name != "PMPI_Alltoallv") ||
      !call.getType()->isIntegerTy(32) || !FP ||
      mpiRealType(call.getArgOperand(7)) != FP || gutils->runtimeActivity ||
      Mode == DerivativeMode::ForwardModeSplit ||
      gutils->isConstantValue(call.getArgOperand(0)) ||
      gutils->isConstantValue(call.getArgOperand(4))) {
    EmitNoDerivativeError(
        "MPI_Alltoallv requires C/PMPI, static float/double "
        "datatypes and active out-of-place buffers; runtime activity and split "
        "forward are unsupported",
        call, gutils, B);
    return true;
  }
  bool Forward = Mode == DerivativeMode::ForwardMode ||
                 Mode == DerivativeMode::ForwardModeError;
  auto Primal = [&](unsigned I) {
    return gutils->getNewFromOriginal(call.getArgOperand(I));
  };
  if (Forward) {
    B.SetInsertPoint(New->getNextNode());
    for (unsigned Lane = 0; Lane < gutils->getWidth(); ++Lane) {
      SmallVector<Value *, 9> Args;
      for (unsigned I = 0; I < 9; ++I)
        Args.push_back(Primal(I));
      for (unsigned I : {0u, 4u}) {
        auto Shadow = gutils->invertPointerM(call.getArgOperand(I), B);
        if (gutils->getWidth() > 1)
          Shadow = gutils->extractMeta(B, Shadow, Lane);
        Args[I] = B.CreatePointerCast(Shadow, Args[I]->getType());
      }
      B.CreateCall(call.getFunctionType(), called, Args);
    }
    return true;
  }
  auto T = alltoallvTape(called);
  Value *Tape;
  if (Mode == DerivativeMode::ReverseModeGradient) {
    Tape = B.CreatePHI(T, 0);
  } else {
    auto Snapshot = alltoallvSnapshot(called);
    SmallVector<Value *, 5> Args;
    for (unsigned I : {1u, 2u, 5u, 6u, 8u})
      Args.push_back(Primal(I));
    for (unsigned I = 0; I < 4; ++I)
      Args[I] = B.CreatePointerCast(Args[I], getUnqual(B.getInt32Ty()));
    auto Saved = B.CreateCall(Snapshot, Args);
    gutils->TapesToPreventRecomputation.insert(Saved);
    Tape = Saved;
  }
  Tape = gutils->cacheForReverse(B, Tape, getIndex(&call, CacheType::Tape, B));
  if (Mode == DerivativeMode::ReverseModePrimal)
    return true;
  getReverseBuilder(B, /*original=*/false);
  Tape = lookup(Tape, B);
  auto DS = lookup(gutils->invertPointerM(call.getArgOperand(0), B), B);
  auto DR = lookup(gutils->invertPointerM(call.getArgOperand(4), B), B);
  for (unsigned Lane = 0; Lane < gutils->getWidth(); ++Lane) {
    auto S = DS, R = DR;
    if (gutils->getWidth() > 1) {
      S = gutils->extractMeta(B, S, Lane);
      R = gutils->extractMeta(B, R, Lane);
    }
    auto Rule =
        alltoallvReverse(called, FP, call.getArgOperand(3),
                         shouldFree() && Lane + 1 == gutils->getWidth());
    B.CreateCall(Rule, {Tape, B.CreatePointerCast(S, getUnqual(FP)),
                        B.CreatePointerCast(R, getUnqual(FP))});
  }
  if (Mode == DerivativeMode::ReverseModeGradient)
    eraseIfUnused(call, /*erase*/ true, /*check*/ false);
  return true;
}
