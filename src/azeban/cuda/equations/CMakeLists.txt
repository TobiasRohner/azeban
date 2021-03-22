if(ZISA_HAS_CUDA)

target_sources(azeban_cuda_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/burgers_cuda.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/incompressible_euler_cuda.cu
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/incompressible_euler_naive_cuda.cu
)

endif()
