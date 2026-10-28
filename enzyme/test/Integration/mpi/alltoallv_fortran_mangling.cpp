// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// REQUIRES: mpi-fortran-abi-cxx, mpi-fortran-abi-double-underscore
// clang-format off
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O1 -fno-exceptions %mpi_cxx_include $flags -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %mpi_fortran_abi_libs -o %t.exe && env OMPI_MCA_rmaps_base_oversubscribe=1 %mpiexec -n 4 %t.exe
// clang-format on
#define DOUBLE_UNDERSCORE
#include "alltoallv_fortran_abi.cpp"
