# Optional legacy Fortran MPI differentiation runtime

This experimental C++23 runtime supports the compiler's opt-in
`--enzyme-mpi-legacy-nonblocking` lowering for legacy Fortran Isend/Irecv/Waitall.
The plugin has no MPI link dependency. This library is linked by generated
programs; its entry points are an internal protocol and must match the compiler
build. Client programs also link their MPI legacy Fortran bindings.

Build through the Enzyme CMake project:

```sh
cmake -S enzyme -B build -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DENZYME_ENABLE_MPI_LEGACY_RUNTIME=ON
cmake --build build --target EnzymeMPILegacyRuntime
cmake --build build --target check-enzyme-mpi-legacy
```

The build option is OFF by default. It requires CMake >=3.20, C++23 and MPI C++
development files. Disable ENZYME_MPI_LEGACY_TESTS to build only the runtime.
The focused test target additionally needs matching clang++/opt, mpiexec and
legacy Fortran bindings. It constructs generated AD programs, verifies their
LLVM IR, links them to this runtime and runs them through CTest. Test sources
exercise exact primal/JVP/VJP values, short-receive padding, existing adjoints,
noinline completion wrappers, loop/request reuse and communicator mutation.
Native lifecycle tests also check request canaries and explicit rejection of
overlapping channels and copied active requests. They do not invoke Python/lit.
ENZYME_MPI_TEST_CXX_FLAGS supplies additional matching-clang SDK flags if needed.
The module also supports a standalone configure with ENZYME_MPI_PLUGIN_FILE.

Runtime completion metadata is keyed by stable primal request-slot address;
Fortran request integers never contain C++ pointers. Completion frees the slot
mapping but keeps each reverse record alive until its reverse initiator finishes.
Reverse Waitall posts the inverse batch; reverse starts wait, accumulate source
adjoints or clear only the actually received portion of a destination shadow.
All MPI buffers, communicators and datatypes must remain live through reversal.

Supported bounded domain: host buffers, one thread, i32 MPI_Fint, one tangent
lane, active buffers, immutable predefined double datatype handles, stable slots
and at most one outstanding operation of each kind per communicator/peer/tag.
No unmanaged traffic may overlap those channels. MPI_F_STATUS_SIZE is required;
execution has been validated on Open MPI only. Wildcards, copied/reordered active
requests, persistent requests, cancellation, device buffers, mpi_f08, independent
reusable reverse tapes and runtime activity are outside this capability. A
standalone MPI_Wait ABI rule is not implemented. Invalid recognized operations
fail with explicit diagnostics where checked; this is not a complete MPI trace
validator. Performance, scaling and indirect completion wrappers are unmeasured.

Compiler metadata-only completion-effect scans follow direct calls and respect
explicit enzyme_inactive annotations. No claim of a general effect system is made.
Existing C Alltoallv work overlaps l90lpa's draft PR #1292 and retains that credit;
the combined private candidate is not claimed as the first C implementation.
The prior scientific replay receipts are separate from these focused tests and
from official upstream CI or a full FMS/MOM6 model execution.
