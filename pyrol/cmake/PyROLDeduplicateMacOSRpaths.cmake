IF(NOT DEFINED PYROL_TARGET_FILE OR NOT EXISTS "${PYROL_TARGET_FILE}")
  MESSAGE(FATAL_ERROR "PYROL_TARGET_FILE must name an existing file.")
ENDIF()

FIND_PROGRAM(PYROL_OTOOL_EXECUTABLE otool)
FIND_PROGRAM(PYROL_INSTALL_NAME_TOOL_EXECUTABLE install_name_tool)

IF(NOT PYROL_OTOOL_EXECUTABLE)
  MESSAGE(FATAL_ERROR "Could not find otool.")
ENDIF()

IF(NOT PYROL_INSTALL_NAME_TOOL_EXECUTABLE)
  MESSAGE(FATAL_ERROR "Could not find install_name_tool.")
ENDIF()

EXECUTE_PROCESS(
  COMMAND "${PYROL_OTOOL_EXECUTABLE}" -l "${PYROL_TARGET_FILE}"
  RESULT_VARIABLE _pyrol_otool_result
  OUTPUT_VARIABLE _pyrol_otool_output
  ERROR_VARIABLE _pyrol_otool_error
)

IF(NOT _pyrol_otool_result EQUAL 0)
  MESSAGE(FATAL_ERROR
    "Could not inspect LC_RPATH entries in ${PYROL_TARGET_FILE}: "
    "${_pyrol_otool_error}")
ENDIF()

STRING(REPLACE "\r\n" "\n" _pyrol_otool_output "${_pyrol_otool_output}")
STRING(REPLACE "\n" ";" _pyrol_otool_lines "${_pyrol_otool_output}")

SET(_pyrol_rpaths "")
FOREACH(_pyrol_line IN LISTS _pyrol_otool_lines)
  STRING(STRIP "${_pyrol_line}" _pyrol_line)
  IF(_pyrol_line MATCHES "^path (.*) \\(offset [0-9]+\\)$")
    LIST(APPEND _pyrol_rpaths "${CMAKE_MATCH_1}")
  ENDIF()
ENDFOREACH()

IF(NOT _pyrol_rpaths)
  RETURN()
ENDIF()

SET(_pyrol_unique_rpaths "")
FOREACH(_pyrol_rpath IN LISTS _pyrol_rpaths)
  IF(NOT _pyrol_rpath IN_LIST _pyrol_unique_rpaths)
    LIST(APPEND _pyrol_unique_rpaths "${_pyrol_rpath}")
  ENDIF()
ENDFOREACH()

LIST(LENGTH _pyrol_rpaths _pyrol_rpath_count)
LIST(LENGTH _pyrol_unique_rpaths _pyrol_unique_rpath_count)

IF(_pyrol_rpath_count EQUAL _pyrol_unique_rpath_count)
  RETURN()
ENDIF()

FOREACH(_pyrol_rpath IN LISTS _pyrol_unique_rpaths)
  WHILE(TRUE)
    EXECUTE_PROCESS(
      COMMAND "${PYROL_INSTALL_NAME_TOOL_EXECUTABLE}"
              -delete_rpath "${_pyrol_rpath}" "${PYROL_TARGET_FILE}"
      RESULT_VARIABLE _pyrol_delete_result
      OUTPUT_QUIET
      ERROR_QUIET
    )

    IF(NOT _pyrol_delete_result EQUAL 0)
      BREAK()
    ENDIF()
  ENDWHILE()
ENDFOREACH()

FOREACH(_pyrol_rpath IN LISTS _pyrol_unique_rpaths)
  EXECUTE_PROCESS(
    COMMAND "${PYROL_INSTALL_NAME_TOOL_EXECUTABLE}"
            -add_rpath "${_pyrol_rpath}" "${PYROL_TARGET_FILE}"
    RESULT_VARIABLE _pyrol_add_result
    ERROR_VARIABLE _pyrol_add_error
  )

  IF(NOT _pyrol_add_result EQUAL 0)
    MESSAGE(FATAL_ERROR
      "Could not add LC_RPATH ${_pyrol_rpath} to ${PYROL_TARGET_FILE}: "
      "${_pyrol_add_error}")
  ENDIF()
ENDFOREACH()

MESSAGE(STATUS
  "Deduplicated macOS LC_RPATH entries for ${PYROL_TARGET_FILE}: "
  "${_pyrol_rpath_count} -> ${_pyrol_unique_rpath_count}")
