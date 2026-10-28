// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
// REQUIRES: mpi-cxx
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O1 -fno-exceptions %mpi_cxx_include $flags  -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %mpi_cxx_libs -o %t.exe && env OMPI_MCA_rmaps_base_oversubscribe=1 %mpiexec -n 4 %t.exe
// RUN: if [ %llvmver -lt 17 ]; then flags="-Xclang -no-opaque-pointers"; else flags=""; fi; %clang++ -std=c++2b -O2 -fno-exceptions %mpi_cxx_include $flags  -S -emit-llvm %s -o %t.input.ll && %opt %loadEnzyme %enzyme -S %t.input.ll -o %t.ad.ll && %clang++ -O2 %t.ad.ll %mpi_cxx_libs -o %t.exe && env OMPI_MCA_rmaps_base_oversubscribe=1 %mpiexec -n 4 %t.exe
// clang-format on

// C++23 end-to-end distributed inverse-problem probe, external MPI backend.
#define OMPI_SKIP_MPICXX 1
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mpi.h>
#include <vector>
int enzyme_const;
extern "C" double __enzyme_fwddiff(void *, ...);
extern "C" void __enzyme_autodiff(void *, ...);

// A nonlinear source field is redistributed to observations on remote ranks.
// The global objective is the sum of each rank's weighted least-squares loss.
__attribute__((noinline)) double objective(double *parameters, double *send,
                                           double *receive, int n, int *sc,
                                           int *sd, int *rc, int *rd, int rank,
                                           int peers) {
  for (int i = 0; i < n; ++i)
    send[i] = parameters[i] * parameters[i];
  MPI_Alltoallv(send, sc, sd, MPI_DOUBLE, receive, rc, rd, MPI_DOUBLE,
                MPI_COMM_WORLD);
  double result = 0;
  for (int p = 0; p < peers; ++p)
    for (int k = 0; k < rc[p]; ++k) {
      double weight = 1 + (rank + p + k) / 8.0;
      double residual = receive[rd[p] + k] - (rank + 1) / 16.0;
      result += weight * residual * residual / 2;
    }
  return result;
}
int count(int source, int target) { return (source + 2 * target) % 3; }
int displacement(int source, int target, bool overlap) {
  if (overlap)
    return target % 2;
  int offset = 0;
  for (int p = 0; p < target; ++p)
    offset += count(source, p) + 1;
  return offset;
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, peers;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &peers);
  bool ok = true;
  for (bool overlap : {false, true}) {
    std::vector<int> sc(peers), sd(peers), rc(peers), rd(peers);
    int ns = 4, nr = 0;
    for (int p = 0; p < peers; ++p) {
      sc[p] = count(rank, p);
      sd[p] = displacement(rank, p, overlap);
      rc[p] = count(p, rank);
      rd[p] = nr;
      ns = std::max(ns, sd[p] + sc[p] + 2);
      nr += rc[p] + 2;
    }
    std::vector<double> x(ns), send(ns), recv(nr), gx(ns, 1.0 / 64), gs(ns, 0),
        gr(nr, 0), direction(ns), analytic(ns, 1.0 / 64);
    for (int i = 0; i < ns; ++i) {
      x[i] = 0.5 + (rank + 1) / 8.0 + i / 32.0;
      direction[i] = (i + 1) / 32.0;
    }
    for (int dest = 0; dest < peers; ++dest)
      for (int k = 0; k < sc[dest]; ++k) {
        int i = sd[dest] + k;
        double weight = 1 + (dest + rank + k) / 8.0;
        analytic[i] += 2 * x[i] * weight * (x[i] * x[i] - (dest + 1) / 16.0);
      }
    __enzyme_autodiff((void *)objective, x.data(), gx.data(), send.data(),
                      gs.data(), recv.data(), gr.data(), enzyme_const, ns,
                      enzyme_const, sc.data(), enzyme_const, sd.data(),
                      enzyme_const, rc.data(), enzyme_const, rd.data(),
                      enzyme_const, rank, enzyme_const, peers);
    double localError = 0, expectedDirection = 0;
    for (int i = 0; i < ns; ++i) {
      localError = std::max(localError, std::abs(gx[i] - analytic[i]));
      expectedDirection += (analytic[i] - 1.0 / 64) * direction[i];
    }
    std::fill(gs.begin(), gs.end(), 0);
    std::fill(gr.begin(), gr.end(), 0);
    double jvp = __enzyme_fwddiff(
        (void *)objective, x.data(), direction.data(), send.data(), gs.data(),
        recv.data(), gr.data(), enzyme_const, ns, enzyme_const, sc.data(),
        enzyme_const, sd.data(), enzyme_const, rc.data(), enzyme_const,
        rd.data(), enzyme_const, rank, enzyme_const, peers);
    auto plus = x, minus = x;
    const double h = 1e-6;
    for (int i = 0; i < ns; ++i) {
      plus[i] += h * direction[i];
      minus[i] -= h * direction[i];
    }
    double fd =
        (objective(plus.data(), send.data(), recv.data(), ns, sc.data(),
                   sd.data(), rc.data(), rd.data(), rank, peers) -
         objective(minus.data(), send.data(), recv.data(), ns, sc.data(),
                   sd.data(), rc.data(), rd.data(), rank, peers)) /
        (2 * h);
    double values[4] = {localError, jvp, expectedDirection, fd}, global[4];
    MPI_Allreduce(values, global, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ok &= global[0] < 1e-11 && std::abs(global[1] - global[2]) < 1e-11 &&
          std::abs(global[1] - global[3]) < 1e-7;
    if (!rank)
      std::printf("ranks=%d overlap=%d gradient_error=%.17g jvp=%.17g "
                  "analytic=%.17g fd=%.17g\n",
                  peers, overlap, global[0], global[1], global[2], global[3]);
  }
  int bad = !ok, total;
  MPI_Allreduce(&bad, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (!rank)
    std::printf("DISTRIBUTED_INVERSE_PASS=%d\n", total == 0);
  MPI_Finalize();
  return total != 0;
}
