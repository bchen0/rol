INCLUDE(PythonTest)

FILE(COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/test_objective.py
  ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/dynamic/navier-stokes/input.xml
  ${${PACKAGE_NAME}_SOURCE_DIR}/example/PDE-OPT/dynamic/navier-stokes/channel.txt
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

PYTHON_ADD_TEST(test_objective)
