# Optional C++23 runtime; the compiler plugin remains MPI-independent.
if(CMAKE_VERSION VERSION_LESS 3.20)
  message(FATAL_ERROR "Legacy MPI runtime requires CMake 3.20 or newer")
endif()
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(EnzymeMPILegacyRuntime LANGUAGES CXX)
  enable_testing()
endif()
find_package(MPI REQUIRED COMPONENTS CXX)
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
check_cxx_source_compiles("#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
static_assert(sizeof(MPI_Fint)==4, \"requires i32 Fortran handles\");
#ifndef MPI_F_STATUS_SIZE
#error requires MPI_F_STATUS_SIZE
#endif
int main() { return MPI_F_STATUS_SIZE < 1; }" ENZYME_MPI_LEGACY_LAYOUT)
cmake_pop_check_state()
if(NOT ENZYME_MPI_LEGACY_LAYOUT)
  message(FATAL_ERROR "Legacy MPI runtime requires i32 MPI_Fint and MPI_F_STATUS_SIZE")
endif()
add_library(EnzymeMPILegacyRuntime STATIC LegacyRuntime.cpp)
target_compile_features(EnzymeMPILegacyRuntime PUBLIC cxx_std_23)
target_compile_definitions(EnzymeMPILegacyRuntime PRIVATE OMPI_SKIP_MPICXX=1)
target_include_directories(EnzymeMPILegacyRuntime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(EnzymeMPILegacyRuntime PUBLIC MPI::MPI_CXX)
install(TARGETS EnzymeMPILegacyRuntime ARCHIVE DESTINATION lib COMPONENT mpi-runtime)
option(ENZYME_MPI_LEGACY_TESTS "Build focused native and generated-AD MPI tests" ON)
if(ENZYME_MPI_LEGACY_TESTS)
  add_subdirectory(tests)
endif()
