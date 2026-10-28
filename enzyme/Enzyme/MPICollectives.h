// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once
namespace llvm {
class Type;
class Value;
} // namespace llvm
// Recognize predefined scalar real handles shared by MPI type analysis and AD.
// The legacy Fortran ABI requires an immutable global i32 handle.
llvm::Type *getMPIAlltoallvRealType(llvm::Value *V, bool FortranABI);
bool isMPIFortranBufferSentinel(llvm::Value *V);

// Experimental generated code requires the separately linked MPI runtime.
bool useLegacyMPINonblocking();
