if(ZISA_HAS_CUDA)

target_sources(azeban_cuda_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/axpy_cuda.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/convolve.cu
)

endif()
