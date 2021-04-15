target_sources(azeban_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/profiler.cpp
)

if(ZISA_HAS_CUDA)

endif()
add_subdirectory(cuda)
add_subdirectory(equations)
add_subdirectory(init)
add_subdirectory(operations)
