target_sources(azeban_unit_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/fft.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/operations.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/initializer.cpp
)

add_subdirectory(equations)
