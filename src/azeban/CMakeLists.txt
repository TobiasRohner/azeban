target_sources(azeban_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/some_source_file.cpp
)

if(ZISA_HAS_CUDA)

endif()
add_subdirectory(cuda)
