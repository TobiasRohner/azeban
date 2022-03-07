target_sources(azeban_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/communicator.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/cuda_aware_communicator.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/manual_cuda_aware_communicator.cpp
)

if(ZISA_HAS_CUDA)

endif()
