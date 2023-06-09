if(ZISA_HAS_CUDA)

target_sources(azeban_cuda_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/cuda_check_error.cpp
)

endif()
add_subdirectory(equations)
add_subdirectory(operations)
add_subdirectory(random)
add_subdirectory(forcing)
