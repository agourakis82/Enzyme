// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// REQUIRES: mpi-fortran-abi-cxx
// clang-format off
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O1 -fno-exceptions %mpi_cxx_include $flags -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %mpi_fortran_abi_libs -o %t.exe && not %mpiexec -n 1 %t.exe 2>&1 | FileCheck %s
// clang-format on
#define OMPI_SKIP_MPICXX 1
#include <cstdio>
#include <mpi.h>
// Negative control: the library's real-width query disagrees with the compiled
// scalar type. Communication still uses real MPI; only this query is injected.
extern "C" void mpi_type_size_(const MPI_Fint *, MPI_Fint *size,
                               MPI_Fint *ierr) {
  *size = 123;
  *ierr = 0;
  std::fprintf(stderr, "INJECTED_FORTRAN_REAL_WIDTH\n");
}
#include "alltoallv_fortran_abi.cpp"
// CHECK: INJECTED_FORTRAN_REAL_WIDTH
