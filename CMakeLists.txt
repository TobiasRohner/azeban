cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(azeban VERSION 0.1 LANGUAGES C CXX)

if(ZISA_HAS_CUDA)
  enable_language(CUDA)
endif()

include("cmake/has_build_type.cmake")
include("cmake/warning_flags.cmake")
include("cmake/fast_debug.cmake")

# -- Dependencies --------------------------------------------------------------
add_library(azeban_dependencies INTERFACE)

find_package(FFTW3 REQUIRED)
target_link_libraries(azeban_dependencies INTERFACE fftw3)

find_package(nlohmann_json 3.9.1 REQUIRED)
target_link_libraries(azeban_dependencies INTERFACE nlohmann_json::nlohmann_json)

find_package(fmt REQUIRED)
target_link_libraries(azeban_dependencies INTERFACE fmt::fmt)

find_package(HDF5 REQUIRED)
target_include_directories(azeban_dependencies INTERFACE ${HDF5_INCLUDE_DIRS})
target_link_libraries(azeban_dependencies INTERFACE ${HDF5_LIBRARIES})

# experimental/filesystem
target_link_libraries(azeban_dependencies INTERFACE stdc++fs)

if(AZEBAN_DO_PROFILE)
  target_compile_definitions(azeban_dependencies INTERFACE AZEBAN_DO_PROFILE=1)
endif()

# Logging
if(ZISA_LOG_LEVEL)
  target_compile_definitions(azeban_dependencies INTERFACE _LOG_LEVEL=${ZISA_LOG_LEVEL})
endif()

if(ZISA_HAS_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(azeban_dependencies INTERFACE ZISA_HAS_CUDA=1)

  target_compile_options(azeban_dependencies INTERFACE
    $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>>
  )
  target_link_libraries(azeban_dependencies INTERFACE CUDA::cudart)
  target_link_libraries(azeban_dependencies INTERFACE CUDA::cufft)
endif()

# -- internal ----------------------------------------------------------------------
find_package(ZisaCore REQUIRED)
target_link_libraries(azeban_dependencies INTERFACE Zisa::core)

find_package(ZisaMemory REQUIRED)
target_link_libraries(azeban_dependencies INTERFACE Zisa::memory)

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

if(ZISA_HAS_CUDA)
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

add_library(azeban_lib STATIC)
set_target_properties(azeban_lib PROPERTIES OUTPUT_NAME azeban)
target_sources(azeban_lib PRIVATE $<TARGET_OBJECTS:azeban_generic_obj>)

if(ZISA_HAS_CUDA)
  target_sources(azeban_lib PRIVATE $<TARGET_OBJECTS:azeban_cuda_obj>)
  set_target_properties(azeban_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

target_include_directories(azeban_lib PUBLIC
        $<BUILD_INTERFACE:${azeban_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
target_link_libraries(azeban_lib PUBLIC azeban_dependencies)
target_link_libraries(azeban_lib PRIVATE azeban_warning_flags)

add_executable(azeban)
target_link_libraries(azeban PRIVATE azeban_lib)
target_link_libraries(azeban PRIVATE azeban_warning_flags)
set_target_properties(azeban PROPERTIES CXX_STANDARD 17)


# -- Tests ---------------------------------------------------------------------
find_package(Catch2 REQUIRED)

add_executable(azeban_unit_tests "")
set_target_properties(azeban_unit_tests PROPERTIES OUTPUT_NAME unit_tests)
set_target_properties(azeban_unit_tests PROPERTIES CXX_STANDARD 17)
if(ZISA_HAS_CUDA)
  set_target_properties(azeban_unit_tests PROPERTIES CUDA_STANDARD 11)
endif()
target_link_libraries(azeban_unit_tests PUBLIC Catch2::Catch2)
target_link_libraries(azeban_unit_tests PUBLIC azeban_lib)
target_include_directories(azeban_unit_tests PRIVATE test)
target_link_libraries(azeban_unit_tests PRIVATE azeban_warning_flags)

enable_testing()
add_test(NAME AzebanUnitTests COMMAND unit_tests)

# -- Micro benchmarks ----------------------------------------------------------
if(AZEBAN_HAS_BENCHMARK)
  find_package(benchmark REQUIRED)

  add_executable(azeban_benchmarks "")
  set_target_properties(azeban_benchmarks PROPERTIES OUTPUT_NAME micro_benchmarks)
  set_target_properties(azeban_benchmarks PROPERTIES CXX_STANDARD 17)
  if(ZISA_HAS_CUDA)
    set_target_properties(azeban_benchmarks PROPERTIES CUDA_STANDARD 11)
  endif()
  target_include_directories(azeban_benchmarks PRIVATE
      $<BUILD_INTERFACE:${azeban_SOURCE_DIR}/benchmarks>
  )
  target_link_libraries(azeban_benchmarks PRIVATE azeban_lib benchmark::benchmark)
endif()

# -- Subdirectories ------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(test)

if(AZEBAN_HAS_BENCHMARK)
  add_subdirectory(benchmarks)
endif()
