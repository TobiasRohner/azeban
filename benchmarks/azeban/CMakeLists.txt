target_sources(azeban_benchmarks
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/fft.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/fft_3d.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/incompressible_euler.cpp
)

if(ZISA_HAS_CUDA)

endif()
