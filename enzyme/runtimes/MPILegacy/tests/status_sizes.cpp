// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#define OMPI_SKIP_MPICXX 1
#include "MPILegacyNonblocking.hpp"
#include <limits>

// The baseline branch repeats the original signed expression, without
// allocating an enormous request/status array or claiming to execute a giant
// MPI operation.
static std::size_t words(MPI_Fint n) {
#ifdef ENZYME_TEST_ORIGINAL_STATUS_ARITHMETIC
  return n * MPI_F_STATUS_SIZE;
#else
  return enzyme::legacy_mpi::detail::statusWordCount(n);
#endif
}
int main() {
  for (MPI_Fint n :
       {MPI_Fint(0), MPI_Fint(1), std::numeric_limits<MPI_Fint>::max()}) {
    std::uint64_t expected = 0;
    for (int i = 0; i < MPI_F_STATUS_SIZE; ++i)
      expected += static_cast<std::uint64_t>(n);
    if (expected > std::numeric_limits<std::size_t>::max())
      continue; // This platform cannot represent that status array.
    volatile MPI_Fint input = n;
    const auto actual = words(input);
    if (static_cast<std::uint64_t>(actual) != expected) {
      std::fprintf(stderr, "STATUS_WORDS_MISMATCH\n");
      return 2;
    }
  }
  std::puts("STATUS_WORDS_BOUNDARY_PASS");
}
