find_program(ENZYME_MPI_CLANGXX NAMES clang++ HINTS "${LLVM_TOOLS_BINARY_DIR}")
find_program(ENZYME_MPI_OPT NAMES opt HINTS "${LLVM_TOOLS_BINARY_DIR}")
find_library(ENZYME_MPI_LEGACY_BINDINGS NAMES mpi_mpifh mpifort)
if(NOT ENZYME_MPI_CLANGXX OR NOT ENZYME_MPI_OPT OR NOT ENZYME_MPI_LEGACY_BINDINGS)
  message(FATAL_ERROR "Legacy MPI tests require matching clang++/opt and legacy Fortran bindings")
endif()
if(TARGET LLVMEnzyme-${LLVM_VERSION_MAJOR})
  set(plugin "$<TARGET_FILE:LLVMEnzyme-${LLVM_VERSION_MAJOR}>")
  set(plugin_dep LLVMEnzyme-${LLVM_VERSION_MAJOR})
elseif(ENZYME_MPI_PLUGIN_FILE AND EXISTS "${ENZYME_MPI_PLUGIN_FILE}")
  set(plugin "${ENZYME_MPI_PLUGIN_FILE}")
  set(plugin_dep "${plugin}")
else()
  message(FATAL_ERROR "Standalone AD tests require ENZYME_MPI_PLUGIN_FILE")
endif()
set(ENZYME_MPI_TEST_CXX_FLAGS "" CACHE STRING "Additional flags for the matching test clang++")
separate_arguments(test_flags NATIVE_COMMAND "${ENZYME_MPI_TEST_CXX_FLAGS}")
set(include_flags)
foreach(dir IN LISTS MPI_CXX_INCLUDE_DIRS)
  list(APPEND include_flags "-I${dir}")
endforeach()
set(test_targets)
foreach(name IN ITEMS direct wrapper loop metadata)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp")
  set(exe "${CMAKE_CURRENT_BINARY_DIR}/${name}.exe")
  set(extra)
  if(name STREQUAL "wrapper")
    set(src "${CMAKE_CURRENT_SOURCE_DIR}/direct.cpp")
    set(extra -DWRAPPER)
  endif()
  add_custom_command(OUTPUT "${exe}"
    COMMAND "${ENZYME_MPI_CLANGXX}" ${test_flags} -std=c++23 -O1
      ${include_flags} ${extra} -S -emit-llvm "${src}" -o "${exe}.input.ll"
    COMMAND "${ENZYME_MPI_OPT}" "-load-pass-plugin=${plugin}" "-load=${plugin}"
      --enzyme-attributor=0 --enzyme-mpi-legacy-nonblocking -passes=enzyme -S
      "${exe}.input.ll" -o "${exe}.ad.ll"
    COMMAND "${ENZYME_MPI_OPT}" -passes=verify -disable-output "${exe}.ad.ll"
    COMMAND "${ENZYME_MPI_CLANGXX}" ${test_flags} -O2 "${exe}.ad.ll"
      "$<TARGET_FILE:EnzymeMPILegacyRuntime>" ${MPI_CXX_LIBRARIES}
      "${ENZYME_MPI_LEGACY_BINDINGS}" -o "${exe}"
    DEPENDS "${src}" EnzymeMPILegacyRuntime ${plugin_dep}
    VERBATIM COMMAND_EXPAND_LISTS)
  add_custom_target(EnzymeMPILegacy-${name} DEPENDS "${exe}")
  list(APPEND test_targets EnzymeMPILegacy-${name})
  foreach(ranks IN ITEMS 1 2 4)
    add_test(NAME EnzymeMPILegacy.${name}.${ranks}
      COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} ${ranks}
        ${MPIEXEC_PREFLAGS} "${exe}" ${MPIEXEC_POSTFLAGS})
    set_tests_properties(EnzymeMPILegacy.${name}.${ranks} PROPERTIES TIMEOUT 45
      ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1")
  endforeach()
endforeach()
add_executable(EnzymeMPILegacyLifecycle lifecycle.cpp)
target_link_libraries(EnzymeMPILegacyLifecycle PRIVATE EnzymeMPILegacyRuntime "${ENZYME_MPI_LEGACY_BINDINGS}")
list(APPEND test_targets EnzymeMPILegacyLifecycle)
add_test(NAME EnzymeMPILegacy.lifecycle
  COMMAND "${MPIEXEC_EXECUTABLE}" -n 2 "$<TARGET_FILE:EnzymeMPILegacyLifecycle>")
foreach(negative IN ITEMS overlap copied)
  if(negative STREQUAL "overlap")
    set(expected "LEGACY_NB_REJECT: overlapping operations on same channel")
  else()
    set(expected "LEGACY_NB_REJECT: untracked request or copied request slot")
  endif()
  add_test(NAME EnzymeMPILegacy.reject.${negative}
    COMMAND "${CMAKE_COMMAND}" "-DMPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DEXE=$<TARGET_FILE:EnzymeMPILegacyLifecycle>" "-DNEGATIVE=${negative}"
      "-DEXPECTED=${expected}" -P "${CMAKE_CURRENT_SOURCE_DIR}/reject.cmake")
endforeach()
set_tests_properties(EnzymeMPILegacy.lifecycle EnzymeMPILegacy.reject.overlap
  EnzymeMPILegacy.reject.copied PROPERTIES TIMEOUT 40
  ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1")
add_executable(EnzymeMPILegacyStatusSizes status_sizes.cpp)
target_link_libraries(EnzymeMPILegacyStatusSizes PRIVATE EnzymeMPILegacyRuntime)
list(APPEND test_targets EnzymeMPILegacyStatusSizes)
add_test(NAME EnzymeMPILegacy.status-sizes COMMAND EnzymeMPILegacyStatusSizes)
add_custom_target(check-enzyme-mpi-legacy
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -R "^EnzymeMPILegacy[.]"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  DEPENDS ${test_targets} USES_TERMINAL)
