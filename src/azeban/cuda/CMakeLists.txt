if(ZISA_HAS_CUDA)

target_sources(azeban_cuda_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/some_kernel.cu
)

endif()
