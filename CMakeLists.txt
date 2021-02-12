cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(azeban VERSION 0.1 LANGUAGES C CXX)

if(_HAS_CUDA)
  enable_language(CUDA)
endif()

include("cmake/has_build_type.cmake")
include("cmake/warning_flags.cmake")
include("cmake/fast_debug.cmake")

# -- Dependencies --------------------------------------------------------------
add_library(azeban_dependencies INTERFACE)

find_package(HDF5 REQUIRED)
target_include_directories(azeban_dependencies INTERFACE ${HDF5_INCLUDE_DIRS})
target_link_libraries(azeban_dependencies INTERFACE ${HDF5_LIBRARIES})

# experimental/filesystem
target_link_libraries(azeban_dependencies INTERFACE stdc++fs)

# Logging
if(_LOG_LEVEL)
  target_compile_definitions(azeban_dependencies INTERFACE _LOG_LEVEL=${ZISA_LOG_LEVEL})
endif()

if(_HAS_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_compile_definitions(azeban_dependencies INTERFACE _HAS_MPI=1)
endif()

if(_HAS_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(azeban_dependencies INTERFACE _HAS_CUDA=1)

  target_compile_options(azeban_dependencies INTERFACE
    $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>>
  )
  target_link_libraries(azeban_dependencies INTERFACE CUDA::cudart)
endif()

# -- internal ----------------------------------------------------------------------
find_package(Core REQUIRED)
target_link_libraries(azeban_dependencies INTERFACE ::core)

find_package(Memory REQUIRED)
target_link_libraries(azeban_dependencies INTERFACE ::memory)

# -- azeban --------------------------------------------------------------------
add_library(azeban_generic_obj OBJECT)
set_target_properties(azeban_generic_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(azeban_generic_obj PROPERTIES CXX_STANDARD 17)
target_include_directories(azeban_generic_obj PUBLIC
  $<BUILD_INTERFACE:${azeban_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )
target_link_libraries(azeban_generic_obj PUBLIC azeban_dependencies)
target_link_libraries(azeban_generic_obj PRIVATE azeban_warning_flags)

if(_HAS_CUDA)
  add_library(azeban_cuda_obj OBJECT)
  set_target_properties(azeban_cuda_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(azeban_cuda_obj PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(azeban_cuda_obj PROPERTIES CUDA_STANDARD 11)
  set_target_properties(azeban_cuda_obj PROPERTIES CXX_STANDARD 17)
  target_include_directories(azeban_cuda_obj PUBLIC
    $<BUILD_INTERFACE:${azeban_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )
  target_link_libraries(azeban_cuda_obj PUBLIC azeban_dependencies)
  target_link_libraries(azeban_cuda_obj PRIVATE azeban_warning_flags)
endif()

add_executable(azeban)
target_sources(azeban PRIVATE $<TARGET_OBJECTS:azeban_generic_obj>)

if(_HAS_CUDA)
  target_sources(azeban PRIVATE $<TARGET_OBJECTS:azeban_cuda_obj>)
  set_target_properties(azeban PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

target_include_directories(azeban PUBLIC
  $<BUILD_INTERFACE:${azeban_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(azeban PUBLIC azeban_dependencies)
target_link_libraries(azeban PUBLIC ::core)
target_link_libraries(azeban PRIVATE azeban_warning_flags)

# -- install -------------------------------------------------------------------
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/azebanConfigVersion.cmake"
  VERSION ${azeban_VERSION}
  COMPATIBILITY SameMajorVersion
  )

install(
  FILES
  "cmake/azebanConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/azebanConfigVersion.cmake"
  DESTINATION lib/cmake/
  )

install(
  TARGETS azeban
          azeban_dependencies
          azeban_warning_flags
  EXPORT azebanTargets
  )
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT azebanTargets
  FILE azebanTargets.cmake
  NAMESPACE ::
  DESTINATION lib/cmake/
  )

# -- Tests ---------------------------------------------------------------------
find_package(Catch2 REQUIRED)

add_executable(azeban_unit_tests "")
set_target_properties(azeban_unit_tests PROPERTIES OUTPUT_NAME unit_tests)
set_target_properties(azeban_unit_tests PROPERTIES CXX_STANDARD 17)
if(_HAS_CUDA)
  set_target_properties(azeban_unit_tests PROPERTIES CUDA_STANDARD 11)
endif()
target_link_libraries(azeban_unit_tests PUBLIC Catch2::Catch2)
target_include_directories(azeban_unit_tests PRIVATE test)
target_link_libraries(azeban_unit_tests PRIVATE azeban_warning_flags)

enable_testing()
add_test(NAME UnitTests COMMAND unit_tests)

# -- Subdirectories ------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(test)
