add_subdirectory(azeban)

target_sources(azeban
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/azeban.cpp
)


target_sources(postprocess
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/postprocess.cpp
)
