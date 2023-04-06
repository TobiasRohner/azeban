target_sources(azeban_generic_obj
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/brownian_motion.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/discontinuous_double_shear_layer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/discontinuous_shear_tube.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/discontinuous_vortex_patch.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/double_shear_layer.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/init_3d_from_2d.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/init_from_file.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/python.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/shear_tube.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/shock.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/sine_1d.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/taylor_green.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/taylor_vortex.cpp
)

if(ZISA_HAS_CUDA)

endif()
