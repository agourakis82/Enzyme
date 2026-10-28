// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
// REQUIRES: mpi-cxx
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O1 -fno-exceptions %mpi_cxx_include $flags  -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %mpi_cxx_libs -o %t.exe && env OMPI_MCA_rmaps_base_oversubscribe=1 %mpiexec -n 4 %t.exe
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O1 -fno-exceptions %mpi_cxx_include $flags -DVECTOR2 -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %mpi_cxx_libs -o %t.exe && env OMPI_MCA_rmaps_base_oversubscribe=1 %mpiexec -n 4 %t.exe
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O1 -fno-exceptions %mpi_cxx_include $flags -DUSE_PMPI -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %mpi_cxx_libs -o %t.exe && env OMPI_MCA_rmaps_base_oversubscribe=1 %mpiexec -n 4 %t.exe
// clang-format on

// C++23 distributed regression: asymmetric traffic, overlapping sends, tape.
#define OMPI_SKIP_MPICXX 1
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <mpi.h>
#include <type_traits>
#include <vector>
int enzyme_const, enzyme_width;
extern "C" void __enzyme_fwddiff(void *, ...);
extern "C" void __enzyme_autodiff(void *, ...);
#ifdef USE_PMPI
#define EXCHANGE PMPI_Alltoallv
#else
#define EXCHANGE MPI_Alltoallv
#endif

template <class T>
__attribute__((noinline)) void exchange(T *send, T *recv, int *sc, int *sd,
                                        int *rc, int *rd, int peers,
                                        bool clobber) {
  if constexpr (std::is_same_v<T, double>)
    EXCHANGE(send, sc, sd, MPI_DOUBLE, recv, rc, rd, MPI_DOUBLE,
             MPI_COMM_WORLD);
  else
    EXCHANGE(send, sc, sd, MPI_FLOAT, recv, rc, rd, MPI_FLOAT, MPI_COMM_WORLD);
  if (clobber)
    for (int i = 0; i < peers; ++i) {
      sc[i] = 0;
      sd[i] = 0;
      rc[i] = 0;
      rd[i] = 0;
    }
}
int count(int s, int d, bool empty) { return empty ? 0 : (s + 2 * d) % 3; }
int recvOffset(int source, int dest, bool empty) {
  int v = 0;
  for (int s = 0; s < source; ++s)
    v += count(s, dest, empty) + 2;
  return v;
}
int sendOffset(int source, int dest, bool empty, bool overlap) {
  if (overlap)
    return dest % 2;
  int v = 0;
  for (int d = 0; d < dest; ++d)
    v += count(source, d, empty) + 1;
  return v;
}
template <class T>
bool test(int rank, int peers, bool empty, bool overlap, bool clobber) {
  std::vector<int> sc(peers), sd(peers), rc(peers), rd(peers);
  int ns = 4, nr = 0;
  for (int p = 0; p < peers; ++p) {
    sc[p] = count(rank, p, empty);
    sd[p] = sendOffset(rank, p, empty, overlap);
    rc[p] = count(p, rank, empty);
    rd[p] = recvOffset(p, rank, empty);
    ns = std::max(ns, sd[p] + sc[p] + 2);
    nr = std::max(nr, rd[p] + rc[p] + 2);
  }
  auto osc = sc, osd = sd, orc = rc, ord = rd;
  std::vector<T> s(ns), r(nr, T(-7)), ds(ns), dr(nr, T(9)), ds1(ns),
      dr1(nr, T(11));
  for (int i = 0; i < ns; ++i) {
    s[i] = T(rank) + T(i) / 16;
    ds[i] = T(rank) + T(i) / 8 + T(.25);
    ds1[i] = -ds[i] / 2;
  }
#ifdef VECTOR2
  __enzyme_fwddiff((void *)exchange<T>, enzyme_width, 2, s.data(), ds.data(),
                   ds1.data(), r.data(), dr.data(), dr1.data(),
#else
  __enzyme_fwddiff((void *)exchange<T>, s.data(), ds.data(), r.data(),
                   dr.data(),
#endif
                   enzyme_const, sc.data(), enzyme_const, sd.data(),
                   enzyme_const, rc.data(), enzyme_const, rd.data(),
                   enzyme_const, peers, enzyme_const, clobber);
  bool ok = true;
  std::vector<bool> used(nr, false);
  for (int p = 0; p < peers; ++p)
    for (int k = 0; k < orc[p]; ++k) {
      int i = ord[p] + k;
      used[i] = true;
      int sourceIndex = sendOffset(p, rank, empty, overlap) + k;
      T primal = T(p) + T(sourceIndex) / 16,
        tangent = T(p) + T(sourceIndex) / 8 + T(.25);
      ok &= r[i] == primal && dr[i] == tangent;
#ifdef VECTOR2
      ok &= dr1[i] == -tangent / 2;
#endif
    }
  for (int i = 0; i < nr; ++i)
    if (!used[i])
      ok &= dr[i] == T(9) && r[i] == T(-7);
  sc = osc;
  sd = osd;
  rc = orc;
  rd = ord;
  std::fill(ds.begin(), ds.end(), T(.5));
  std::fill(ds1.begin(), ds1.end(), T(.25));
  std::vector<T> expected(ns, T(.5)), expected1(ns, T(.25));
  for (int i = 0; i < nr; ++i) {
    dr[i] = T(2 * rank) + T(i) / 8 + T(.5);
    dr1[i] = -dr[i] / 2;
  }
  auto before = dr;
  for (int p = 0; p < peers; ++p)
    for (int k = 0; k < osc[p]; ++k) {
      T g = T(2 * p) + T(recvOffset(rank, p, empty) + k) / 8 + T(.5);
      expected[osd[p] + k] += g;
      expected1[osd[p] + k] -= g / 2;
    }
#ifdef VECTOR2
  __enzyme_autodiff((void *)exchange<T>, enzyme_width, 2, s.data(), ds.data(),
                    ds1.data(), r.data(), dr.data(), dr1.data(),
#else
  __enzyme_autodiff((void *)exchange<T>, s.data(), ds.data(), r.data(),
                    dr.data(),
#endif
                    enzyme_const, sc.data(), enzyme_const, sd.data(),
                    enzyme_const, rc.data(), enzyme_const, rd.data(),
                    enzyme_const, peers, enzyme_const, clobber);
  for (int i = 0; i < ns; ++i) {
    ok &= ds[i] == expected[i];
#ifdef VECTOR2
    ok &= ds1[i] == expected1[i];
#endif
  }
  for (int i = 0; i < nr; ++i) {
    ok &= dr[i] == (used[i] ? T(0) : before[i]);
#ifdef VECTOR2
    ok &= dr1[i] == (used[i] ? T(0) : -before[i] / 2);
#endif
  }
  if (!ok)
    std::fprintf(stderr,
                 "FAIL rank=%d float=%d empty=%d overlap=%d clobber=%d\n", rank,
                 std::is_same_v<T, float>, empty, overlap, clobber);
  return ok;
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, peers;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &peers);
  bool ok = true;
  for (bool empty : {false, true})
    for (bool overlap : {false, true})
      for (bool clobber : {false, true}) {
        ok &= test<float>(rank, peers, empty, overlap, clobber);
        ok &= test<double>(rank, peers, empty, overlap, clobber);
      }
  int bad = !ok, total = 0;
  MPI_Allreduce(&bad, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (!rank)
    std::printf("MPI_IRREGULAR_STRESS_PASS=%d ranks=%d\n", total == 0, peers);
  MPI_Finalize();
  return total != 0;
}
