; RUN: not %opt < %s %loadEnzyme %enzyme -S 2>&1 | FileCheck %s
; A primal in-place collective is valid MPI, but is outside this rule's scope.
target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"
@ompi_mpi_double = external global i8
@ompi_mpi_comm_world = external global i8

declare i32 @MPI_Alltoallv(i8*, i32*, i32*, i8*, double*, i32*, i32*, i8*, i8*)
declare void @__enzyme_fwddiff(...)

define void @exchange(double* %recv, i32* %counts, i32* %displs) {
  %status = call i32 @MPI_Alltoallv(i8* inttoptr(i64 1 to i8*),
      i32* %counts, i32* %displs, i8* @ompi_mpi_double,
      double* %recv, i32* %counts, i32* %displs,
      i8* @ompi_mpi_double, i8* @ompi_mpi_comm_world)
  ret void
}

define void @differentiate(double* %recv, double* %shadow, i32* %counts, i32* %displs) {
  call void (...) @__enzyme_fwddiff(void (double*, i32*, i32*)* @exchange,
                                  double* %recv, double* %shadow,
                                  i32* %counts, i32* %counts,
                                  i32* %displs, i32* %displs)
  ret void
}
; CHECK: MPI_Alltoallv requires C/PMPI, static float/double datatypes and active out-of-place buffers
