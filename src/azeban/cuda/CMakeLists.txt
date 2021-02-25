if(ZISA_HAS_CUDA)

target_sources(azeban_cuda_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/copy_padded_cuda.cu
)

endif()
add_subdirectory(operations)
