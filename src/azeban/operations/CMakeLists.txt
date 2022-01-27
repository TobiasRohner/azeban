target_sources(azeban_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/copy_to_padded.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/fft.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/fft_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/leray.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/transpose.cpp
)

if(ZISA_HAS_CUDA)

endif()
