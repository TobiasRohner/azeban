target_sources(azeban_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/profiler.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/simulation.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/simulation_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/simulation_mpi_factory.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/run_from_config.cpp
)

if(ZISA_HAS_CUDA)

endif()
add_subdirectory(cuda)
add_subdirectory(equations)
add_subdirectory(init)
add_subdirectory(io)
add_subdirectory(memory)
add_subdirectory(mpi)
add_subdirectory(operations)
