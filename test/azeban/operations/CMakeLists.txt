target_sources(azeban_unit_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/axpy.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/fft.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/norm.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/leray.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/copy_to_padded.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/copy_from_padded.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/inverse_curl.cpp
)
