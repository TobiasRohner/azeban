target_sources(azeban_unit_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/burgers.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/incompressible_euler.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/incompressible_euler_naive.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/incompressible_euler_optimized_correctness.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/spectral_viscosity.cpp
)

