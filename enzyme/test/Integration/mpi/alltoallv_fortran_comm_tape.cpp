// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// REQUIRES: mpi-fortran-abi-cxx
// clang-format off
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O1 -fno-exceptions %mpi_cxx_include $flags -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %mpi_fortran_abi_libs -o %t.exe && env OMPI_MCA_rmaps_base_oversubscribe=1 %mpiexec -n 4 %t.exe
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O2 -fno-exceptions %mpi_cxx_include $flags -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %mpi_fortran_abi_libs -o %t.exe && env OMPI_MCA_rmaps_base_oversubscribe=1 %mpiexec -n 4 %t.exe
// clang-format on
#define OMPI_SKIP_MPICXX 1
#include <cstdio>
#include <mpi.h>
#include <vector>
static_assert(sizeof(MPI_Fint) == 4);
#if defined(OMPI_MAJOR_VERSION)
static const MPI_Fint dtype = 17;
#elif defined(MPICH)
static const MPI_Fint dtype = 0x4c00081f;
#endif
extern "C" void mpi_alltoallv_(const void *, const MPI_Fint *, const MPI_Fint *,
                               const MPI_Fint *, void *, const MPI_Fint *,
                               const MPI_Fint *, const MPI_Fint *,
                               const MPI_Fint *, MPI_Fint *);
int enzyme_const;
extern "C" void __enzyme_fwddiff(void *, ...);
extern "C" void __enzyme_autodiff(void *, ...);
__attribute__((noinline)) void exchange(double *x, double *b, int *counts,
                                        int *sd, int *rd, MPI_Fint *comm) {
  MPI_Fint ierr = -1;
  mpi_alltoallv_(x, counts, sd, &dtype, b, counts, rd, &dtype, comm, &ierr);
  // Caller-visible mutation survives optimization before AD. Saving this
  // address on the tape instead of its value would invalidate reverse MPI.
  *comm = -1;
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, peers;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &peers);
  MPI_Fint comm = MPI_Comm_c2f(MPI_COMM_WORLD), saved = comm;
  std::vector<int> counts(peers, 1), sd(peers, 0), rd(peers);
  for (int p = 0; p < peers; ++p)
    rd[p] = 2 * p + 1;
  int n = 2 * peers + 2;
  double x[2]{double(rank + 3), 7}, dx[2]{double(rank + 2), -1}, bar[2]{7, 7};
  std::vector<double> b(n, -3), db(n, 5), reversePrimal(n, -3), seed(n);
  for (int i = 0; i < n; ++i)
    seed[i] = rank + i + 1;
  __enzyme_fwddiff((void *)exchange, x, dx, b.data(), db.data(), enzyme_const,
                   counts.data(), enzyme_const, sd.data(), enzyme_const,
                   rd.data(), enzyme_const, &comm);
  bool ok = comm == -1;
  comm = saved;
  __enzyme_autodiff((void *)exchange, x, bar, reversePrimal.data(), seed.data(),
                    enzyme_const, counts.data(), enzyme_const, sd.data(),
                    enzyme_const, rd.data(), enzyme_const, &comm);
  ok &= comm == -1;
  double expected = 7;
  for (int p = 0; p < peers; ++p)
    expected += p + 2 * rank + 2;
  ok &= bar[0] == expected && bar[1] == 7;
  for (int i = 0; i < n; ++i) {
    bool written = i % 2 == 1 && i < 2 * peers;
    int source = (i - 1) / 2;
    ok &= b[i] == (written ? source + 3 : -3);
    ok &= reversePrimal[i] == b[i];
    ok &= db[i] == (written ? source + 2 : 5);
    ok &= seed[i] == (written ? 0 : rank + i + 1);
  }
  int local = ok, global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!global)
    MPI_Abort(MPI_COMM_WORLD, 2);
  if (!rank)
    std::printf("FORTRAN_COMM_VALUE_TAPE_PASS ranks=%d\n", peers);
  MPI_Finalize();
  return 0;
}
