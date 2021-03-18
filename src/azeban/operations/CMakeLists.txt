target_sources(azeban_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/copy_padded.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/fft.cpp
)

if(ZISA_HAS_CUDA)

endif()
