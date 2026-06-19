IF(PYROL_HAS_GENERATED_BINDINGS)
FILE(GLOB PYROL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
MESSAGE("PYROL_SRC = ${PYROL_SRC}")
FILE(COPY ${PYROL_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

MESSAGE("CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")

# file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/../binder/pyrol.sources BINDER_SRCS)
list(TRANSFORM BINDER_SRCS PREPEND "${CMAKE_CURRENT_BINARY_DIR}/../binder/")
list(APPEND PYROL_SRC ${BINDER_SRCS})

list(APPEND PYROL_SRC ${CMAKE_CURRENT_BINARY_DIR}/../binder/pyrol.cpp)

IF(NOT PYROL_USE_ONE_FILE)
  MATH(EXPR NUMBER_FILE "${PYROL_BINDER_NUM_FILES}")

  foreach(index RANGE 0 ${NUMBER_FILE})
    list(APPEND PYROL_SRC ${CMAKE_CURRENT_BINARY_DIR}/../binder/pyrol_${index}.cpp)
  endforeach()

  MATH(EXPR NUMBER_FILE "${NUMBER_FILE}+1")

  EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}"
    -D "NUMBER_FILE=${NUMBER_FILE}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/checkNumberFiles.cmake"
    RESULT_VARIABLE STATUS
    OUTPUT_VARIABLE OUTPUT_CHECKNUMBERFILES
  )

  if(STATUS AND NOT STATUS EQUAL 0)
    message("${OUTPUT_CHECKNUMBERFILES}")
    message(FATAL_ERROR "checkNumberFiles FAILED: ${STATUS}")
  endif()
ENDIF()

MESSAGE("PYROL_SRC with Binder = ${PYROL_SRC}")

pybind11_add_module(pyrol ${PYROL_SRC})
target_include_directories(pyrol PUBLIC ${Mpi4Py_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/../binder "$<BUILD_INTERFACE:${ROL_all_include_dirs}>")
target_compile_features(pyrol PUBLIC cxx_std_14)

foreach(depPkg IN LISTS ROL_LIB_ENABLED_DEPENDENCIES)
    target_link_libraries(pyrol PUBLIC ${depPkg}::all_libs)
endforeach()
target_link_libraries(pyrol PUBLIC ${Trilinos_EXTRA_LINK_FLAGS})
set_target_properties(pyrol PROPERTIES SUFFIX ".so")

INSTALL(TARGETS pyrol
  DESTINATION ${PyROL_INSTALL_DIR})

add_custom_command(TARGET pyrol POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/pyrol.so ${CMAKE_CURRENT_BINARY_DIR}/../pyrol/.
    COMMENT "Copy ${PROJECT_BINARY_DIR}/src/pyrol.so"
)
ENDIF()

IF(PYROL_ENABLE_NAVIER_STOKES)
  SET(PYROL_NAVIER_STOKES_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/navier_stokes/PyROL_NavierStokes.cpp
    ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/TOOLS/solver.cpp
    ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/TOOLS/assembler.cpp
  )

  SET(PYROL_NAVIER_STOKES_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/navier_stokes
    ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/TOOLS
    ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/dynamic/navier-stokes
    ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/teuchos/src/sol
    ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/tpetra/src/vector
    ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/tpetra/src/sol
  )

  IF(TPL_ENABLE_MPI)
    LIST(APPEND PYROL_NAVIER_STOKES_INCLUDE_DIRS
      ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/mpi/src/function
      ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/mpi/src/vector
      ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/tpetra/src/mpi
    )
  ENDIF()

  pybind11_add_module(_navier_stokes ${PYROL_NAVIER_STOKES_SRC})
  target_include_directories(_navier_stokes PUBLIC
    ${Mpi4Py_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYROL_NAVIER_STOKES_INCLUDE_DIRS}
    "$<BUILD_INTERFACE:${ROL_all_include_dirs}>"
  )
  target_compile_features(_navier_stokes PUBLIC cxx_std_14)

  foreach(depPkg IN LISTS ROL_LIB_ENABLED_DEPENDENCIES)
    target_link_libraries(_navier_stokes PUBLIC ${depPkg}::all_libs)
  endforeach()
  target_link_libraries(_navier_stokes PUBLIC Intrepid::all_libs)
  target_link_libraries(_navier_stokes PUBLIC ${Trilinos_EXTRA_LINK_FLAGS})
  set_target_properties(_navier_stokes PROPERTIES SUFFIX ".so")

  INSTALL(TARGETS _navier_stokes
    DESTINATION ${PyROL_INSTALL_DIR})

  add_custom_command(TARGET _navier_stokes POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_navier_stokes> ${CMAKE_CURRENT_BINARY_DIR}/../pyrol/.
    COMMENT "Copy $<TARGET_FILE:_navier_stokes>")

  SET(PYROL_NAVIER_STOKES_REFERENCE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/navier_stokes/PyROL_NavierStokes_Reference.cpp
    ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/TOOLS/solver.cpp
    ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/TOOLS/assembler.cpp
  )

  add_executable(navier_stokes_reference ${PYROL_NAVIER_STOKES_REFERENCE_SRC})
  target_include_directories(navier_stokes_reference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYROL_NAVIER_STOKES_INCLUDE_DIRS}
    "$<BUILD_INTERFACE:${ROL_all_include_dirs}>"
  )
  target_compile_features(navier_stokes_reference PUBLIC cxx_std_14)

  foreach(depPkg IN LISTS ROL_LIB_ENABLED_DEPENDENCIES)
    target_link_libraries(navier_stokes_reference PUBLIC ${depPkg}::all_libs)
  endforeach()
  target_link_libraries(navier_stokes_reference PUBLIC Intrepid::all_libs)
  target_link_libraries(navier_stokes_reference PUBLIC ${Trilinos_EXTRA_LINK_FLAGS})

  add_custom_command(TARGET navier_stokes_reference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:navier_stokes_reference> ${CMAKE_CURRENT_BINARY_DIR}/../pyrol/.
    COMMENT "Copy $<TARGET_FILE:navier_stokes_reference>")
ENDIF()

IF(PYROL_ENABLE_FLOW_OPT)
  SET(PYROL_FLOW_OPT_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/flow_opt
    ${${PACKAGE_NAME}_SOURCE_DIR}
    ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/TOOLS
    ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/teuchos/src/sol
    ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/tpetra/src/function
    ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/tpetra/src/vector
    ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/tpetra/src/sol
  )

  IF(TPL_ENABLE_MPI)
    LIST(APPEND PYROL_FLOW_OPT_INCLUDE_DIRS
      ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/mpi/src/function
      ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/mpi/src/vector
      ${${PACKAGE_NAME}_SOURCE_DIR}/adapters/tpetra/src/mpi
    )
  ENDIF()

  FUNCTION(PYROL_ADD_FLOW_OPT_MODULE MODULE_NAME MODULE_SOURCE)
    SET(PYROL_FLOW_OPT_SRC
      ${CMAKE_CURRENT_SOURCE_DIR}/flow_opt/${MODULE_SOURCE}
      ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/TOOLS/solver.cpp
      ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/TOOLS/assembler.cpp
    )

    pybind11_add_module(${MODULE_NAME} ${PYROL_FLOW_OPT_SRC})
    target_include_directories(${MODULE_NAME} PUBLIC
      ${Mpi4Py_INCLUDE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${PYROL_FLOW_OPT_INCLUDE_DIRS}
      "$<BUILD_INTERFACE:${ROL_all_include_dirs}>"
    )
    target_compile_features(${MODULE_NAME} PUBLIC cxx_std_14)

    foreach(depPkg IN LISTS ROL_LIB_ENABLED_DEPENDENCIES)
      target_link_libraries(${MODULE_NAME} PUBLIC ${depPkg}::all_libs)
    endforeach()
    target_link_libraries(${MODULE_NAME} PUBLIC Intrepid::all_libs)
    target_link_libraries(${MODULE_NAME} PUBLIC ${Trilinos_EXTRA_LINK_FLAGS})
    set_target_properties(${MODULE_NAME} PROPERTIES SUFFIX ".so")

    INSTALL(TARGETS ${MODULE_NAME}
      DESTINATION ${PyROL_INSTALL_DIR})

    add_custom_command(TARGET ${MODULE_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${MODULE_NAME}> ${CMAKE_CURRENT_BINARY_DIR}/../pyrol/.
      COMMENT "Copy $<TARGET_FILE:${MODULE_NAME}>")
  ENDFUNCTION()

  PYROL_ADD_FLOW_OPT_MODULE(_flow_opt_darcy PyROL_FlowOpt_Darcy.cpp)
  PYROL_ADD_FLOW_OPT_MODULE(_flow_opt_brinkman PyROL_FlowOpt_Brinkman.cpp)
  PYROL_ADD_FLOW_OPT_MODULE(_flow_opt_filtered_darcy PyROL_FlowOpt_FilteredDarcy.cpp)

  FUNCTION(PYROL_ADD_FLOW_OPT_REFERENCE TARGET_NAME TARGET_SOURCE)
    SET(PYROL_FLOW_OPT_REFERENCE_SRC
      ${CMAKE_CURRENT_SOURCE_DIR}/flow_opt/${TARGET_SOURCE}
      ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/TOOLS/solver.cpp
      ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/TOOLS/assembler.cpp
    )

    add_executable(${TARGET_NAME} ${PYROL_FLOW_OPT_REFERENCE_SRC})
    target_include_directories(${TARGET_NAME} PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${PYROL_FLOW_OPT_INCLUDE_DIRS}
      "$<BUILD_INTERFACE:${ROL_all_include_dirs}>"
    )
    target_compile_features(${TARGET_NAME} PUBLIC cxx_std_14)

    foreach(depPkg IN LISTS ROL_LIB_ENABLED_DEPENDENCIES)
      target_link_libraries(${TARGET_NAME} PUBLIC ${depPkg}::all_libs)
    endforeach()
    target_link_libraries(${TARGET_NAME} PUBLIC Intrepid::all_libs)
    target_link_libraries(${TARGET_NAME} PUBLIC ${Trilinos_EXTRA_LINK_FLAGS})

    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> ${CMAKE_CURRENT_BINARY_DIR}/../pyrol/.
      COMMENT "Copy $<TARGET_FILE:${TARGET_NAME}>")
  ENDFUNCTION()

  PYROL_ADD_FLOW_OPT_REFERENCE(flow_opt_darcy_reference PyROL_FlowOpt_Reference_Darcy.cpp)
  PYROL_ADD_FLOW_OPT_REFERENCE(flow_opt_brinkman_reference PyROL_FlowOpt_Reference_Brinkman.cpp)
  PYROL_ADD_FLOW_OPT_REFERENCE(flow_opt_filtered_darcy_reference PyROL_FlowOpt_Reference_FilteredDarcy.cpp)
ENDIF()
