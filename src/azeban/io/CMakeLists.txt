target_sources(azeban_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/writer_collection.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_file.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_file_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_sample_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_energy_spectrum_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_enstrophy_spectrum_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_snapshot_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_snapshot_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_collective_snapshot_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_collective_snapshot_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/paraview_catalyst_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/paraview_catalyst_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/energy_spectrum_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/energy_spectrum_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/enstrophy_spectrum_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/enstrophy_spectrum_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/structure_function_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/structure_function_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/structure_function_cube_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/structure_function_cube_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/second_order_structure_function_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/second_order_structure_function_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_second_order_structure_function_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/netcdf_structure_function_cube_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/third_order_structure_function_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/third_order_structure_function_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/longitudinal_structure_function_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/longitudinal_structure_function_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/absolute_longitudinal_structure_function_writer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/absolute_longitudinal_structure_function_writer_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/writer_factory.cpp
)

if(ZISA_HAS_CUDA)

endif()
