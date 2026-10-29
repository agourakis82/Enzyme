// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Experimental lifecycle used by compiler-generated legacy MPI AD calls.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <mpi.h>
#include <set>
#include <tuple>
#include <vector>

namespace enzyme::legacy_mpi {
static_assert(sizeof(MPI_Fint) == sizeof(std::int32_t));
using Start = void (*)(void *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *,
                       MPI_Fint *, MPI_Fint *, MPI_Fint *);
using WaitAll = void (*)(MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *);
[[noreturn]] inline void fail(const char *why) {
  std::fprintf(stderr, "LEGACY_NB_REJECT: %s\n", why);
  MPI_Abort(MPI_COMM_WORLD, 91);
  std::abort();
}
inline void checked(int rc) {
  if (rc != MPI_SUCCESS)
    fail("MPI error");
}
namespace detail {
inline std::size_t statusWordCount(MPI_Fint count) {
  static_assert(MPI_F_STATUS_SIZE > 0);
  if (count < 0)
    fail("negative wait count");
  const auto n = static_cast<std::size_t>(count);
  if (n > std::numeric_limits<std::size_t>::max() / MPI_F_STATUS_SIZE)
    fail("status storage size overflow");
  return n * static_cast<std::size_t>(MPI_F_STATUS_SIZE);
}
} // namespace detail
enum class Mode { Forward, Reverse };
enum class Kind { Send, Receive };
struct Record {
  Kind kind;
  Mode mode;
  MPI_Fint count, datatype, peer, tag, comm;
  MPI_Fint *slot;
  MPI_Request original, auxiliary = MPI_REQUEST_NULL;
  double *shadow;
  std::vector<double> scratch;
  int actual = 0;
  bool complete = false, reversing = false;
};
// The prototype requires MPI calls from one thread, stable request slots,
// and at most one live operation of each kind on a communicator/peer/tag.
// Request values never contain runtime pointers.
class Lifecycle {
  using Channel = std::tuple<MPI_Fint, MPI_Fint, MPI_Fint, Kind>;
  std::map<MPI_Fint *, Record *> pending;
  std::set<Record *> live;
  std::set<Channel> channels, reverseChannels;
  static Channel channel(const Record &r) {
    return {r.comm, r.peer, r.tag, r.kind};
  }
  static MPI_Comm communicator(const Record &r) { return MPI_Comm_f2c(r.comm); }
  static MPI_Datatype datatype(const Record &r) {
    return MPI_Type_f2c(r.datatype);
  }
  static void post(Record &r, bool reverse) {
    if (r.peer == MPI_PROC_NULL)
      return;
    const int n = reverse ? r.actual : r.count;
    const bool send = reverse ? r.kind == Kind::Receive : r.kind == Kind::Send;
    double *buffer = r.shadow;
    if (reverse && r.kind == Kind::Send) {
      r.scratch.assign(n, 0.0);
      buffer = r.scratch.data();
    }
    if (send)
      checked(MPI_Isend(buffer, n, datatype(r), r.peer, r.tag, communicator(r),
                        &r.auxiliary));
    else
      checked(MPI_Irecv(buffer, n, datatype(r), r.peer, r.tag, communicator(r),
                        &r.auxiliary));
  }

public:
  using Batch = std::vector<Record *>;
  Record *start(Start primal, Kind kind, Mode mode, double *buffer,
                double *shadow, MPI_Fint count, MPI_Fint type, MPI_Fint peer,
                MPI_Fint tag, MPI_Fint comm, MPI_Fint *slot, MPI_Fint *ierr) {
    if (count < 0 || peer == MPI_ANY_SOURCE || tag == MPI_ANY_TAG)
      fail("negative count or wildcard");
    if (pending.contains(slot))
      fail("request slot reused before completion");
    int bytes;
    MPI_Aint lb, extent;
    checked(MPI_Type_size(MPI_Type_f2c(type), &bytes));
    checked(MPI_Type_get_extent(MPI_Type_f2c(type), &lb, &extent));
    if (bytes != sizeof(double) || lb != 0 || extent != sizeof(double) ||
        (MPI_Type_f2c(type) != MPI_DOUBLE &&
         MPI_Type_f2c(type) != MPI_DOUBLE_PRECISION))
      fail("prototype requires predefined double precision");
    auto *r = new Record{kind,
                         mode,
                         count,
                         type,
                         peer,
                         tag,
                         comm,
                         slot,
                         MPI_REQUEST_NULL,
                         MPI_REQUEST_NULL,
                         shadow,
                         {}};
    if (!channels.insert(channel(*r)).second)
      fail("overlapping operations on same channel");
    primal(buffer, &count, &type, &peer, &tag, &comm, slot, ierr);
    checked(*ierr);
    r->original = MPI_Request_f2c(*slot);
    pending.emplace(slot, r);
    live.insert(r);
    if (mode == Mode::Forward)
      post(*r, false);
    return r;
  }
  Batch wait(WaitAll primal, MPI_Fint n, MPI_Fint *slots, MPI_Fint *statuses,
             MPI_Fint *ierr) {
    const auto words = detail::statusWordCount(n);
    Batch batch;
    for (MPI_Fint i = 0; i < n; ++i) {
      auto it = pending.find(slots + i);
      if (it == pending.end()) {
        if (MPI_Request_f2c(slots[i]) != MPI_REQUEST_NULL)
          fail("untracked request or copied request slot");
        batch.push_back(nullptr);
        continue;
      }
      Record *r = it->second;
      if (MPI_Request_f2c(slots[i]) != r->original)
        fail("request value changed");
      batch.push_back(r);
    }
    std::vector<MPI_Fint> scratch(words ? words : detail::statusWordCount(1));
    primal(&n, slots, scratch.data(), ierr);
    checked(*ierr);
    if (statuses != MPI_F_STATUSES_IGNORE)
      for (std::size_t i = 0; i < words; ++i)
        statuses[i] = scratch[i];
    Batch reverse;
    for (MPI_Fint i = 0; i < n; ++i) {
      Record *r = batch[i];
      if (!r)
        continue;
      MPI_Status status;
      checked(MPI_Status_f2c(
          scratch.data() + static_cast<std::size_t>(i) *
                               static_cast<std::size_t>(MPI_F_STATUS_SIZE),
          &status));
      int cancelled;
      checked(MPI_Test_cancelled(&status, &cancelled));
      if (cancelled)
        fail("cancelled request");
      r->actual = r->count;
      if (r->kind == Kind::Receive) {
        checked(MPI_Get_count(&status, datatype(*r), &r->actual));
        if (r->actual < 0 || r->actual > r->count)
          fail("invalid receive count");
      }
      r->complete = true;
      pending.erase(r->slot);
      if (r->mode == Mode::Forward) {
        checked(MPI_Wait(&r->auxiliary, MPI_STATUS_IGNORE));
        channels.erase(channel(*r));
        live.erase(r);
        delete r;
      } else {
        channels.erase(channel(*r));
        reverse.push_back(r);
      }
    }
    return reverse;
  }
  void reverseWait(const Batch &batch) {
    // Post the entire inverse batch before any reverse initiator waits.
    for (Record *r : batch) {
      if (!live.contains(r) || !r->complete || r->reversing)
        fail("invalid reverse completion order");
      if (!reverseChannels.insert(channel(*r)).second)
        fail("overlapping reverse channel");
      r->reversing = true;
      post(*r, true);
    }
  }
  void reverseStart(Record *r) {
    if (!live.contains(r) || !r->reversing)
      fail("missing differentiated completion");
    checked(MPI_Wait(&r->auxiliary, MPI_STATUS_IGNORE));
    if (r->peer != MPI_PROC_NULL) {
      if (r->kind == Kind::Send) {
        for (int i = 0; i < r->actual; ++i)
          r->shadow[i] += r->scratch[i];
      } else {
#ifdef ENZYME_NB_MUTATE_CLEAR_CAPACITY
        const int clear = r->count;
#else
        const int clear = r->actual;
#endif
        for (int i = 0; i < clear; ++i)
          r->shadow[i] = 0.0;
      }
    }
    reverseChannels.erase(channel(*r));
    live.erase(r);
    delete r;
  }
  std::size_t liveCount() const { return live.size(); }
  std::size_t pendingCount() const { return pending.size(); }
};
} // namespace enzyme::legacy_mpi
