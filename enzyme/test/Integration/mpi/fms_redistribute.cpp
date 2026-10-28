// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
// REQUIRES: mpi-cxx
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O1 -fno-exceptions %mpi_cxx_include $flags -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %mpi_cxx_libs -o %t.exe && env OMPI_MCA_rmaps_base_oversubscribe=1 %mpiexec -n 4 %t.exe
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O2 -fno-exceptions %mpi_cxx_include $flags -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %mpi_cxx_libs -o %t.exe && env OMPI_MCA_rmaps_base_oversubscribe=1 %mpiexec -n 4 %t.exe
// clang-format on
// Consumer regression inspired by the field/chunk/storage traversal of FMS
// MPP_DO_REDISTRIBUTE_3D_ (NOAA-GFDL/FMS, Apache-2.0), revision
// 0f0a92443b10e254992f7466548e60bbee8b5cb4, mpp/include/mpp_do_redistribute.fh.
// This standalone collective adapter does not invoke FMS or its Fortran ABI.
#define OMPI_SKIP_MPICXX 1
#include <algorithm>
#include <array>
#include <cstdio>
#include <mpi.h>
int enzyme_const;
extern "C" void __enzyme_fwddiff(void *, ...);
extern "C" void __enzyme_autodiff(void *, ...);
constexpr int slots = 96;
struct Block { int shape[3], lo[3], hi[3]; };
struct Plan { Block send[4], recv[4]; int enabled[4]; };
// Fixed, discrete storage bounds are prepared before AD. The differentiated
// routine still packs, communicates and writes the full output fields.
__attribute__((noinline)) void redistribute(double *x, double *b,
    const Plan *p, int *sc, int *sd, int *rc, int *rd) {
  double send[192]{}, recv[192]{};
  for (int peer = 0; peer < 4; ++peer) {
    if (!p->enabled[peer]) continue;
    const auto &r = p->send[peer]; int pos = sd[peer];
    for (int field = 0; field < 2; ++field)
      for (int chunk = 0; chunk < field + 1; ++chunk)
        for (int z = r.lo[2]; z < r.hi[2]; ++z)
          for (int y = r.lo[1]; y < r.hi[1]; ++y)
            for (int x1 = r.lo[0]; x1 < r.hi[0]; ++x1) {
              int offset = (field ? 32 : 3) + chunk * 24 +
                           x1 + r.shape[0] * (y + r.shape[1] * z);
              send[pos++] = x[offset];
            }
  }
  MPI_Alltoallv(send, sc, sd, MPI_DOUBLE, recv, rc, rd, MPI_DOUBLE,
                MPI_COMM_WORLD);
  for (int peer = 0; peer < 4; ++peer) {
    if (!p->enabled[peer]) continue;
    const auto &r = p->recv[peer]; int pos = rd[peer];
    for (int field = 0; field < 2; ++field)
      for (int chunk = 0; chunk < field + 1; ++chunk)
        for (int z = r.lo[2]; z < r.hi[2]; ++z)
          for (int y = r.lo[1]; y < r.hi[1]; ++y)
            for (int x1 = r.lo[0]; x1 < r.hi[0]; ++x1) {
              int offset = (field ? 32 : 3) + chunk * 24 +
                           x1 + r.shape[0] * (y + r.shape[1] * z);
              b[offset] = recv[pos++];
            }
  }
}
// Reference indexing is expressed in logical coordinates; it does not reuse
// the storage traversal, pack buffers or communication result.
int address(std::array<int, 3> axis, bool output, int x, int y, int z) {
  int logicalShape[3]{output ? 3 : 4, output ? 4 : 3, 2};
  int logical[3]{x, y, z}, shape[3], coord[3];
  for (int a = 0; a < 3; ++a) {
    shape[axis[a]] = logicalShape[a]; coord[axis[a]] = logical[a];
  }
  return coord[2] * shape[0] * shape[1] + coord[1] * shape[0] + coord[0];
}
double value(int which, int rank, int i) {
  int n = rank * slots + i;
  switch (which) {
  case 0: return (7 * n + 3) % 61 - 30;
  case 1: return (3 * n + 1) % 13 - 6;
  case 2: return (n + 5) % 17 - 8;
  case 3: return (5 * n + 2) % 19 - 9;
  case 4: return (11 * n + 7) % 23 - 11;
  default: return (n + 2) % 7 - 3;
  }
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv); int rank, peers;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &peers);
  if (peers != 4) MPI_Abort(MPI_COMM_WORLD, 1);
  bool ok = true; int cases = 0; std::array<int, 3> axis{0, 1, 2};
  do {
    Plan p{}; int sc[4]{}, sd[4]{}, rc[4]{}, rd[4]{};
    auto enabled = [&](int s, int d) { return axis[0] != 2 || (s != 3 && d != 3); };
    for (int peer = 0; peer < 4; ++peer) {
      p.enabled[peer] = enabled(rank, peer);
      sc[peer] = rc[peer] = p.enabled[peer] ? 12 : 0;
      int inShape[3]{4, 3, 2}, outShape[3]{3, 4, 2};
      int inLo[3]{2 * (peer % 2), 1, 0}, outLo[3]{0, peer, 0}, extent[3]{2, 1, 2};
      for (int a = 0; a < 3; ++a) {
        int d = axis[a]; p.send[peer].shape[d] = inShape[a]; p.recv[peer].shape[d] = outShape[a];
        p.send[peer].lo[d] = inLo[a]; p.send[peer].hi[d] = inLo[a] + extent[a];
        p.recv[peer].lo[d] = outLo[a]; p.recv[peer].hi[d] = outLo[a] + extent[a];
      }
      if (peer) { sd[peer] = sd[peer-1] + sc[peer-1]; rd[peer] = rd[peer-1] + rc[peer-1]; }
    }
    std::array<std::array<double, slots>, 6> v, expected;
    for (int f = 0; f < 6; ++f) for (int i = 0; i < slots; ++i) v[f][i] = value(f, rank, i);
    expected = v;
    for (int s = 0; s < 4; ++s) for (int d = 0; d < 4; ++d) {
      if (!enabled(s, d)) continue;
      for (int f = 0; f < 2; ++f) for (int c = 0; c < f + 1; ++c)
        for (int x = 0; x < 2; ++x) for (int z = 0; z < 2; ++z) {
          int base = (f ? 32 : 3) + c * 24;
          int si = base + address(axis, false, 2 * (d % 2) + x, 1, z);
          int ri = base + address(axis, true, x, s, z);
          if (d == rank) { expected[2][ri] = value(0, s, si); expected[3][ri] = value(1, s, si); expected[4][ri] = 0; }
          if (s == rank) expected[5][si] += value(4, d, ri);
        }
    }
    auto primal = v[2], tangent = v[3], reversePrimal = v[2], sendbar = v[5], recvbar = v[4];
    __enzyme_fwddiff((void *)redistribute, v[0].data(), v[1].data(), primal.data(), tangent.data(), enzyme_const, &p, enzyme_const, sc, enzyme_const, sd, enzyme_const, rc, enzyme_const, rd);
    __enzyme_autodiff((void *)redistribute, v[0].data(), sendbar.data(), reversePrimal.data(), recvbar.data(), enzyme_const, &p, enzyme_const, sc, enzyme_const, sd, enzyme_const, rc, enzyme_const, rd);
    ok &= primal == expected[2] && reversePrimal == expected[2] && tangent == expected[3] && sendbar == expected[5] && recvbar == expected[4] && v[0] == expected[0] && v[1] == expected[1];
    ++cases;
  } while (std::next_permutation(axis.begin(), axis.end()));
  int local = ok && cases == 6, global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!global) MPI_Abort(MPI_COMM_WORLD, 2);
  if (!rank) std::printf("FMS_REDIS_DERIVATIVE_PASS cases=%d\n", cases);
  MPI_Finalize(); return 0;
}
