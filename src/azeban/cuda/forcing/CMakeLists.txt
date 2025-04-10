if(ZISA_HAS_CUDA)

target_sources(azeban_cuda_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/white_noise_cuda.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/white_noise_high_freq_cuda.cu
)

endif()
