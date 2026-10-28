// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#define OMPI_SKIP_MPICXX 1
#include "MPILegacyNonblocking.hpp"
using namespace enzyme::legacy_mpi;
static Lifecycle lifecycle;
extern "C" void *__enzyme_legacy_nb_start(Start primal, double *buffer,
                                          double *shadow, MPI_Fint *count,
                                          MPI_Fint *type, MPI_Fint *peer,
                                          MPI_Fint *tag, MPI_Fint *comm,
                                          MPI_Fint *request, MPI_Fint *ierr,
                                          int kind, int reverse) {
  return lifecycle.start(primal, kind ? Kind::Receive : Kind::Send,
                         reverse ? Mode::Reverse : Mode::Forward, buffer,
                         shadow, *count, *type, *peer, *tag, *comm, request,
                         ierr);
}
extern "C" void *__enzyme_legacy_nb_waitall(WaitAll primal, MPI_Fint *count,
                                            MPI_Fint *requests,
                                            MPI_Fint *statuses, MPI_Fint *ierr,
                                            int reverse) {
  auto batch = lifecycle.wait(primal, *count, requests, statuses, ierr);
  return reverse ? new Lifecycle::Batch(std::move(batch)) : nullptr;
}
extern "C" void __enzyme_legacy_nb_reverse_wait(void *opaque) {
  auto *batch = static_cast<Lifecycle::Batch *>(opaque);
  lifecycle.reverseWait(*batch);
  delete batch;
}
extern "C" void __enzyme_legacy_nb_reverse_start(void *opaque) {
  lifecycle.reverseStart(static_cast<Record *>(opaque));
}
extern "C" int __enzyme_legacy_nb_empty() {
  return lifecycle.liveCount() == 0 && lifecycle.pendingCount() == 0;
}
