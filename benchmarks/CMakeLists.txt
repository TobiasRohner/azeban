target_sources(azeban_benchmarks
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/micro_benchmarks.cpp
)

target_sources(system_properties
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/system_properties.cpp
)

if(ZISA_HAS_CUDA)

endif()
add_subdirectory(azeban)
