if(ZISA_HAS_CUDA)

target_sources(azeban_cuda_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/axpby_cuda.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/axpy_cuda.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/convolve.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/copy_to_padded_cuda.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/copy_from_padded_cuda.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/cufft.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/cufft_mpi.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/inverse_curl_cuda.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/leray_cuda.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/norm_cuda.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/scale_cuda.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/transpose.cu
)

endif()
