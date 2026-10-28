execute_process(COMMAND "${MPIEXEC}" -n 2 "${EXE}" "${NEGATIVE}"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
if(NOT "${result}" MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "Expected explicit rejection, got ${result}: ${output}${error}")
endif()
if(NOT "${output}${error}" MATCHES "${EXPECTED}")
  message(FATAL_ERROR "Missing rejection diagnostic: ${output}${error}")
endif()
