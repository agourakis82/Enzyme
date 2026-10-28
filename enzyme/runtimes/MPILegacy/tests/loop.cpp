// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#define OMPI_SKIP_MPICXX 1
#include <cstdio>
#include <mpi.h>
extern "C" {
void mpi_isend_(void *, MPI_Fint *, const MPI_Fint *, MPI_Fint *, MPI_Fint *,
                MPI_Fint *, MPI_Fint *, MPI_Fint *);
void mpi_irecv_(void *, MPI_Fint *, const MPI_Fint *, MPI_Fint *, MPI_Fint *,
                MPI_Fint *, MPI_Fint *, MPI_Fint *);
void mpi_waitall_(MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *);
void __enzyme_fwddiff(void *, ...);
void __enzyme_autodiff(void *, ...);
int __enzyme_legacy_nb_empty();
}
int enzyme_const;
static const MPI_Fint dtype = 17;
__attribute__((noinline)) void finish(MPI_Fint *requests) {
  MPI_Fint n = 2, ierr;
  mpi_waitall_(&n, requests, MPI_F_STATUSES_IGNORE, &ierr);
}
__attribute__((noinline)) void exchange(double *x, double *b, MPI_Fint *comm,
                                        int next, int prev, int phases) {
  MPI_Fint req[2], ierr, receive = 4, send = 2;
  for (int phase = 0; phase < phases; ++phase) {
    MPI_Fint tag = 211 + phase;
    mpi_irecv_(b + 4 * phase, &receive, &dtype, &prev, &tag, comm, req, &ierr);
    mpi_isend_(x, &send, &dtype, &next, &tag, comm, req + 1, &ierr);
#ifdef WRAPPER
    finish(req);
#else
    MPI_Fint n = 2;
    mpi_waitall_(&n, req, MPI_F_STATUSES_IGNORE, &ierr);
#endif
  }
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int next = (rank + 1) % size, prev = (rank + size - 1) % size;
  MPI_Fint comm = MPI_Comm_c2f(MPI_COMM_WORLD);
  int phases = size > 0 ? 2 : 1;
  double x[2] = {double(rank + 1), double(rank + 2)},
         dx[2] = {double(rank + 3), double(rank + 4)};
  double b[8] = {-3, -4, 77, 88, -3, -4, 77, 88},
         db[8] = {-5, -6, 11, 12, -5, -6, 11, 12};
  __enzyme_fwddiff((void *)exchange, x, dx, b, db, enzyme_const, &comm,
                   enzyme_const, next, enzyme_const, prev, enzyme_const,
                   phases);
  bool ok = b[0] == prev + 1 && b[1] == prev + 2 && b[2] == 77 && b[3] == 88 &&
            db[0] == prev + 3 && db[1] == prev + 4 && db[2] == 11 &&
            db[3] == 12 && __enzyme_legacy_nb_empty();
  double bar[2] = {7, 8}, rb[8] = {-3, -4, 77, 88, -3, -4, 77, 88},
         seed[8] = {double(rank + 5), double(rank + 6), 11, 12,
                    double(rank + 7), double(rank + 8), 15, 16};
  __enzyme_autodiff((void *)exchange, x, bar, rb, seed, enzyme_const, &comm,
                    enzyme_const, next, enzyme_const, prev, enzyme_const,
                    phases);
  ok &= bar[0] == 7 + next + 5 + next + 7 &&
        bar[1] == 8 + next + 6 + next + 8 && seed[0] == 0 && seed[1] == 0 &&
        seed[2] == 11 && seed[3] == 12 && rb[0] == prev + 1 &&
        rb[1] == prev + 2 && rb[2] == 77 && rb[3] == 88 &&
        __enzyme_legacy_nb_empty();
  ok &= b[4] == prev + 1 && b[5] == prev + 2 && b[6] == 77 && b[7] == 88 &&
        db[4] == prev + 3 && db[5] == prev + 4 && db[6] == 11 && db[7] == 12;
  ok &= rb[4] == prev + 1 && rb[5] == prev + 2 && rb[6] == 77 && rb[7] == 88 &&
        seed[4] == 0 && seed[5] == 0 && seed[6] == 15 && seed[7] == 16;
  int local = ok, global;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!global)
    MPI_Abort(MPI_COMM_WORLD, 7);
  if (!rank)
    std::printf("LEGACY_NB_GENERATED_LOOP_PASS ranks=%d\n", size);
  MPI_Finalize();
}
