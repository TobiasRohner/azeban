target_sources(azeban_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/writer_collection.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_snapshot_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_snapshot_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/paraview_catalyst_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/paraview_catalyst_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/energy_spectrum_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/energy_spectrum_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/writer_factory.cpp
)

if(ZISA_HAS_CUDA)

endif()
