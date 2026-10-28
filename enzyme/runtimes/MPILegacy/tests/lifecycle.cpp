// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "MPILegacyNonblocking.hpp"
#include <cstring>
extern "C" {
void mpi_isend_(void *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *,
                MPI_Fint *, MPI_Fint *, MPI_Fint *);
void mpi_irecv_(void *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *,
                MPI_Fint *, MPI_Fint *, MPI_Fint *);
void mpi_waitall_(MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *);
}
using namespace enzyme::legacy_mpi;
struct GuardedRequests {
  MPI_Fint before = 1234567;
  MPI_Fint slots[2];
  MPI_Fint after = 7654321;
};
void expect(bool good, const char *why) {
  if (!good)
    fail(why);
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int next = (rank + 1) % size, prev = (rank + size - 1) % size;
  const MPI_Fint comm = MPI_Comm_c2f(MPI_COMM_WORLD),
                 type = MPI_Type_c2f(MPI_DOUBLE);
  Lifecycle life;
  GuardedRequests requests;
  MPI_Fint ierr = 0;
  for (int phase = 0; phase < 3; ++phase) {
    // Reuse the same Fortran integer slots only after completion.
    double send[2] = {double(rank + 1), double(rank + 2)},
           recv[4] = {-8, -8, 77, 88};
    double ds[2] = {double(rank + 3), double(rank + 4)},
           dr[4] = {-9, -9, 11, 12};
    const Mode mode = phase == 0 ? Mode::Forward : Mode::Reverse;
    auto *r = life.start(mpi_irecv_, Kind::Receive, mode, recv, dr, 4, type,
                         prev, 70 + phase, comm, &requests.slots[0], &ierr);
    auto *s = life.start(mpi_isend_, Kind::Send, mode, send, ds, 2, type, next,
                         70 + phase, comm, &requests.slots[1], &ierr);
    if (argc > 1 && !std::strcmp(argv[1], "overlap")) {
      MPI_Fint extra;
      life.start(mpi_isend_, Kind::Send, mode, send, ds, 2, type, next,
                 70 + phase, comm, &extra, &ierr);
    }
    if (argc > 1 && !std::strcmp(argv[1], "copied")) {
      MPI_Fint copied[2] = {requests.slots[0], requests.slots[1]};
      life.wait(mpi_waitall_, 2, copied, MPI_F_STATUSES_IGNORE, &ierr);
    }
    MPI_Fint statuses[2 * MPI_F_STATUS_SIZE];
    auto batch =
        life.wait(mpi_waitall_, 2, requests.slots,
                  phase == 1 ? MPI_F_STATUSES_IGNORE : statuses, &ierr);
    expect(requests.before == 1234567 && requests.after == 7654321,
           "Fortran request canary overwritten");
    expect(MPI_Request_f2c(requests.slots[0]) == MPI_REQUEST_NULL &&
               MPI_Request_f2c(requests.slots[1]) == MPI_REQUEST_NULL,
           "request not completed");
    expect(recv[0] == prev + 1 && recv[1] == prev + 2 && recv[2] == 77 &&
               recv[3] == 88,
           "primal or padding mismatch");
    if (mode == Mode::Forward) {
      expect(dr[0] == prev + 3 && dr[1] == prev + 4 && dr[2] == 11 &&
                 dr[3] == 12,
             "JVP or padding mismatch");
    } else {
      // Incoming adjoints are rank-dependent; initial send adjoints must
      // accumulate.
      dr[0] = rank + 5;
      dr[1] = rank + 6;
      life.reverseWait(batch);
      life.reverseStart(s);
      life.reverseStart(r);
      expect(ds[0] == rank + 3 + next + 5 && ds[1] == rank + 4 + next + 6,
             "VJP accumulation mismatch");
      expect(dr[0] == 0 && dr[1] == 0 && dr[2] == 11 && dr[3] == 12,
             "short receive adjoint preservation");
    }
    expect(life.liveCount() == 0 && life.pendingCount() == 0,
           "runtime records leaked");
  }

  // Retain two reverse tapes while reusing the original request array.
  double sends[2][2] = {{1, 2}, {3, 4}},
         recvs[2][4] = {{0, 0, 77, 88}, {0, 0, 77, 88}};
  double sendbars[2][2] = {{7, 8}, {9, 10}},
         recvbars[2][4] = {{5, 6, 11, 12}, {13, 14, 15, 16}};
  Lifecycle::Batch batches[2];
  Record *rr[2], *ss[2];
  for (int p = 0; p < 2; ++p) {
    rr[p] = life.start(mpi_irecv_, Kind::Receive, Mode::Reverse, recvs[p],
                       recvbars[p], 4, type, prev, 90 + p, comm,
                       &requests.slots[0], &ierr);
    ss[p] =
        life.start(mpi_isend_, Kind::Send, Mode::Reverse, sends[p], sendbars[p],
                   2, type, next, 90 + p, comm, &requests.slots[1], &ierr);
    batches[p] = life.wait(mpi_waitall_, 2, requests.slots,
                           MPI_F_STATUSES_IGNORE, &ierr);
    expect(life.liveCount() == std::size_t(2 * (p + 1)),
           "retained tape lost during slot reuse");
  }
  for (int p = 1; p >= 0; --p) {
    life.reverseWait(batches[p]);
    life.reverseStart(ss[p]);
    life.reverseStart(rr[p]);
    expect(sendbars[p][0] == (p ? 22 : 12) && sendbars[p][1] == (p ? 24 : 14),
           "retained tape accumulation");
    expect(recvbars[p][0] == 0 && recvbars[p][1] == 0 &&
               recvbars[p][2] == (p ? 15 : 11) &&
               recvbars[p][3] == (p ? 16 : 12),
           "retained tape padding");
  }
  for (Mode mode : {Mode::Forward, Mode::Reverse}) {
    for (bool nullPeer : {false, true}) {
      const int count = nullPeer ? 2 : 0;
      double x[2] = {3, 4}, y[2] = {5, 6}, dx[2] = {7, 8}, dy[2] = {9, 10};
      auto *r = life.start(mpi_irecv_, Kind::Receive, mode, y, dy, count, type,
                           nullPeer ? MPI_PROC_NULL : prev, 101, comm,
                           &requests.slots[0], &ierr);
      auto *s = life.start(mpi_isend_, Kind::Send, mode, x, dx, count, type,
                           nullPeer ? MPI_PROC_NULL : next, 101, comm,
                           &requests.slots[1], &ierr);
      auto batch = life.wait(mpi_waitall_, 2, requests.slots,
                             MPI_F_STATUSES_IGNORE, &ierr);
      if (mode == Mode::Reverse) {
        life.reverseWait(batch);
        life.reverseStart(s);
        life.reverseStart(r);
      }
      expect(y[0] == 5 && y[1] == 6 && dy[0] == 9 && dy[1] == 10 &&
                 dx[0] == 7 && dx[1] == 8,
             "zero or PROC_NULL touched state");
    }
  }
  MPI_Fint empty = MPI_Request_c2f(MPI_REQUEST_NULL);
  auto emptyBatch =
      life.wait(mpi_waitall_, 1, &empty, MPI_F_STATUSES_IGNORE, &ierr);
  expect(emptyBatch.empty() && life.liveCount() == 0 &&
             life.pendingCount() == 0,
         "final lifecycle state");
  if (!rank)
    std::printf("LEGACY_NB_LIFECYCLE_PASS ranks=%d phases=3\n", size);
  MPI_Finalize();
}
