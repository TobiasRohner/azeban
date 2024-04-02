target_sources(azeban_benchmarks
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/micro_benchmarks.cpp
)
target_sources(system_properties
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/system_properties.cpp
)

if(ZISA_HAS_CUDA)
  target_sources(benchmark_fft
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/benchmark_fft.cpp
  )
endif()
add_subdirectory(azeban)
