INCLUDE(PythonTest)

FILE(MAKE_DIRECTORY
  ${CMAKE_CURRENT_BINARY_DIR}/models/darcy
  ${CMAKE_CURRENT_BINARY_DIR}/models/brinkman
  ${CMAKE_CURRENT_BINARY_DIR}/models/filteredDarcy)

FILE(COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/test_objectives.py
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

FILE(COPY
  ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/flow-opt/axisymmetric/models/darcy/input.xml
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/models/darcy)

FILE(COPY
  ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/flow-opt/axisymmetric/models/brinkman/input.xml
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/models/brinkman)

FILE(COPY
  ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/flow-opt/axisymmetric/models/filteredDarcy/input.xml
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/models/filteredDarcy)

FILE(COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

PYTHON_ADD_TEST(test_objectives)
