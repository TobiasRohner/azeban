target_sources(azeban_unit_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/initializer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/utils.cpp
)

add_subdirectory(equations)
add_subdirectory(operations)
