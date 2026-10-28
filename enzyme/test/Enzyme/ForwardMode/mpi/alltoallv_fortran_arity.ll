; RUN: not %opt < %s %loadEnzyme %enzyme -S 2>&1 | FileCheck %s
target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"
@dtype = internal constant i32 17
@comm = internal global i32 0
declare void @mpi_alltoallv_(double*, i32*, i32*, i32*, double*, i32*, i32*, i32*, i32*)
declare void @__enzyme_fwddiff(...)
define void @exchange(double* %send, double* %recv, i32* %counts, i32* %displs) {
  %known = load double, double* %send
  store double %known, double* %recv
  %ierr = alloca i32
  call void @mpi_alltoallv_(double* %send, i32* %counts, i32* %displs, i32* @dtype,
    double* %recv, i32* %counts, i32* %displs, i32* @dtype, i32* @comm)
  ret void
}
define void @differentiate(double* %send, double* %dsend, double* %recv, double* %drecv, i32* %counts, i32* %displs) {
  call void (...) @__enzyme_fwddiff(void (double*, double*, i32*, i32*)* @exchange,
    double* %send, double* %dsend, double* %recv, double* %drecv,
    i32* %counts, i32* %counts, i32* %displs, i32* %displs)
  ret void
}
; CHECK: MPI_Alltoallv legacy Fortran requires immutable i32 scalar real datatype handles and active out-of-place buffers
