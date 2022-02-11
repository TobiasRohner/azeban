target_sources(azeban_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/snapshot_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_snapshot_writer.cpp
)

if(ZISA_HAS_CUDA)

endif()
