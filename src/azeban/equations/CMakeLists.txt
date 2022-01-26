target_sources(azeban_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/incompressible_euler_mpi.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/equation_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/equation_mpi_factory.cpp
)

if(ZISA_HAS_CUDA)

endif()
