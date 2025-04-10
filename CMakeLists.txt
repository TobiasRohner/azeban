cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(azeban VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0175 OLD)
set(CMAKE_POLICY_DEFAULT_CMP0097 NEW)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

option(ENABLE_CUDA "Enable CUDA Acceleration" ON)
option(ENABLE_MPI "Enable scaling to multiple nodes using MPI" ON)
option(ENABLE_CUDA_AWARE_MPI "Enable CUDA aware MPI if available" ON)
option(ENABLE_PYTHON "Enable python interface for Initializers" OFF)
option(ENABLE_BENCHMARKS "Compile Benchmarks" ON)
option(ENABLE_PROFILING "Profile he code while running" ON)
option(ENABLE_INSITU "Enable in-situ visualization with catalyt" OFF)
option(SINGLE_PRECISION "Use single instead of double precision arithmetic" OFF)
option(HAVE_HDF5 "Compile with HDF5 support" OFF)
option(HAVE_NETCDF "Compile with NetCDF support" ON)

if(ENABLE_MPI AND NOT ENABLE_CUDA)
  message(FATAL_ERROR "MPI without CUDA is not supported at the moment")
endif()

set(ZISA_HAS_CUDA ${ENABLE_CUDA})
set(AZEBAN_HAS_MPI ${ENABLE_MPI})
set(AZEBAN_HAS_CUDA_AWARE_MPI ${ENABLE_CUDA_AWARE_MPI})
set(AZEBAN_HAS_PYTHON ${ENABLE_PYTHON})
set(AZEBAN_HAS_BENCHMARK ${ENABLE_BENCHMARKS})
set(AZEBAN_DO_PROFILE ${ENABLE_PROFILING})
set(ZISA_HAS_HDF5 ${HAVE_HDF5})
set(ZISA_HAS_NETCDF ${HAVE_NETCDF})
if (SINGLE_PRECISION)
  set(FP_PRECISION "float")
else()
  set(FP_PRECISION "double")
endif()

if(ZISA_HAS_CUDA)
  enable_language(CUDA)
endif()

set(BOOST_INCLUDE_LIBRARIES program_options)
FetchContent_Declare(
  Boost
  GIT_REPOSITORY https://github.com/boostorg/boost.git
  GIT_TAG boost-1.85.0
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.2.1
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.7
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
  OVERRIDE_FIND_PACKAGE
)
if (AZEBAN_HAS_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG master
    OVERRIDE_FIND_PACKAGE
  )
endif()
if (ZISA_HAS_NETCDF)
  if (ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C CXX)
    include_directories(${MPI_C_INCLUDE_PATH})
    include_directories(${MPI_CXX_INCLUDE_PATH})
    add_compile_options($<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${MPI_CXX_COMPILE_FLAGS}>>)
    link_libraries(${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
  endif()
  set(HDF5_PREFER_PARALLEL ON)
  set(BUILD_UTILITIES OFF)
  set(ENABLE_EXAMPLES OFF)
  set(ENABLE_TESTS OFF)
  set(ENABLE_DAP OFF)
  set(ENABLE_HDF5 ON)
  set(HDF5_PARALLEL ON)
  set(ENABLE_NETCDF_4 ON)
  set(ENABLE_PNETCDF OFF)
  set(ENABLE_PARALLEL4 ON)
  set(ENABLE_PARALLEL ON)
  set(USE_PARALLEL ON)
  FetchContent_Declare(
    netcdf
    GIT_REPOSITORY https://github.com/Unidata/netcdf-c.git
    GIT_TAG v4.9.3-rc1
    OVERRIDE_FIND_PACKAGE
  )
endif()

FetchContent_MakeAvailable(Boost nlohmann_json Catch2 googletest benchmark)
if (AZEBAN_HAS_PYTHON)
  FetchContent_MakeAvailable(pybind11)
endif()
if (ZISA_HAS_NETCDF)
  FetchContent_MakeAvailable(netcdf)
  add_library(NetCDF::NetCDF ALIAS netcdf)
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/azeban/config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/azeban/config.hpp
  @ONLY
)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

include("cmake/has_build_type.cmake")
include("cmake/warning_flags.cmake")
include("cmake/fast_debug.cmake")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# -- Dependencies --------------------------------------------------------------
add_library(azeban_dependencies INTERFACE)

if(AZEBAN_HAS_MPI)
  #find_package(MPI REQUIRED COMPONENTS CXX)
  target_compile_definitions(azeban_dependencies INTERFACE 
    $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:AZEBAN_HAS_MPI=1>>)
  if(ZISA_HAS_CUDA AND AZEBAN_HAS_CUDA_AWARE_MPI)
    target_compile_definitions(azeban_dependencies INTERFACE 
      $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:AZEBAN_HAS_CUDA_AWARE_MPI=1>>)
  endif()
  target_link_libraries(azeban_dependencies INTERFACE MPI::MPI_C)
  target_link_libraries(azeban_dependencies INTERFACE MPI::MPI_CXX)
  #target_include_directories(azeban_dependencies INTERFACE ${MPI_C_INCLUDE_PATH})
  #target_include_directories(azeban_dependencies INTERFACE ${MPI_CXX_INCLUDE_PATH})
  #target_compile_options(azeban_dependencies INTERFACE
  #$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${MPI_CXX_COMPILE_FLAGS}>>)
  #target_link_libraries(azeban_dependencies INTERFACE ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
endif()

if(AZEBAN_HAS_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11 REQUIRED)
  target_compile_definitions(azeban_dependencies INTERFACE AZEBAN_HAS_PYTHON=1)
  target_include_directories(azeban_dependencies INTERFACE 
    $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${pybind11_INCLUDE_DIR}>>)
  target_link_libraries(azeban_dependencies INTERFACE Python::Python)
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(azeban_dependencies INTERFACE OpenMP::OpenMP_CXX)

find_package(FFTW3 REQUIRED)
target_include_directories(azeban_dependencies INTERFACE ${FFTW3_INCLUDE_DIRS})
target_link_libraries(azeban_dependencies INTERFACE fftw3f fftw3)

find_package(nlohmann_json 3.9.1 REQUIRED)
target_link_libraries(azeban_dependencies INTERFACE nlohmann_json::nlohmann_json)

find_package(fmt REQUIRED)
target_link_libraries(azeban_dependencies INTERFACE fmt::fmt)

find_package(Boost COMPONENTS program_options REQUIRED)
target_link_libraries(azeban_dependencies INTERFACE Boost::program_options)

if(ENABLE_INSITU)
  target_compile_definitions(azeban_dependencies INTERFACE AZEBAN_HAS_CATALYST=1)
  find_package(catalyst 2.0 REQUIRED)
  target_link_libraries(azeban_dependencies INTERFACE catalyst::catalyst)
else()
  target_compile_definitions(azeban_dependencies INTERFACE AZEBAN_HAS_CATALYST=0)
endif()

if(ZISA_HAS_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS CXX)
  target_include_directories(azeban_dependencies INTERFACE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(azeban_dependencies INTERFACE ${HDF5_LIBRARIES})
  target_compile_definitions(azeban_dependencies INTERFACE ZISA_HAS_HDF5=1)
endif()

if(ZISA_HAS_NETCDF)
  find_package(NetCDF REQUIRED)
  #target_include_directories(azeban_dependencies INTERFACE ${NetCDF_INCLUDE_DIRS})
  #target_link_libraries(azeban_dependencies INTERFACE ${NetCDF_LIBRARIES})
  target_link_libraries(azeban_dependencies INTERFACE NetCDF::NetCDF)
  target_compile_definitions(azeban_dependencies INTERFACE ZISA_HAS_NETCDF=1)
endif()

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
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(azeban_dependencies INTERFACE
      $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:-g -G>>
    )
  endif()
  #target_link_libraries(azeban_dependencies INTERFACE CUDA::cudart)
  target_link_libraries(azeban_dependencies INTERFACE CUDA::cufft)
  target_link_libraries(azeban_dependencies INTERFACE CUDA::curand)
endif()

# -- internal ----------------------------------------------------------------------
FetchContent_Declare(
  ZisaCore
  GIT_REPOSITORY https://github.com/1uc/ZisaCore.git
  GIT_TAG 9243c391b70f5b1ff8c0275905351158c4f59eab
  GIT_SUBMODULES ""
  GIT_SUBMODULES_RECURSE OFF
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  ZisaMemory
  GIT_REPOSITORY https://github.com/1uc/ZisaMemory.git
  GIT_TAG 321013db1830fb77589e13139400aeaea8796f79
  GIT_SUBMODULES ""
  GIT_SUBMODULES_RECURSE OFF
  OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(ZisaCore ZisaMemory)

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
target_link_libraries(azeban_unit_tests PRIVATE Catch2::Catch2)
target_link_libraries(azeban_unit_tests PRIVATE azeban_lib)
target_include_directories(azeban_unit_tests PRIVATE test)
target_link_libraries(azeban_unit_tests PRIVATE azeban_warning_flags)
#target_link_libraries(azeban_unit_tests PRIVATE MPI::MPI_CXX)

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

  add_executable(system_properties)
  set_target_properties(system_properties PROPERTIES CXX_STANDARD 17)
  if(ZISA_HAS_CUDA)
    set_target_properties(system_properties PROPERTIES CUDA_STANDARD 11)
    target_compile_definitions(system_properties PUBLIC ZISA_HAS_CUDA=1)
  endif()
  if(AZEBAN_HAS_MPI)
    target_compile_definitions(system_properties PUBLIC AZEBAN_HAS_MPI=1)
    if(ZISA_HAS_CUDA AND AZEBAN_HAS_CUDA_AWARE_MPI)
      target_compile_definitions(system_properties PUBLIC AZEBAN_HAS_CUDA_AWARE_MPI=1)
    endif()
    target_include_directories(system_properties PUBLIC ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(system_properties PUBLIC ${MPI_CXX_COMPILE_FLAGS})
    target_link_libraries(system_properties PUBLIC ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
  endif()
  target_link_libraries(system_properties PRIVATE fmt::fmt)
  target_link_libraries(system_properties PRIVATE azeban_lib)
  target_link_libraries(system_properties PRIVATE azeban_warning_flags)

  add_executable(benchmark_fft)
  set_target_properties(benchmark_fft PROPERTIES CXX_STANDARD 17)
  if(ZISA_HAS_CUDA)
    set_target_properties(benchmark_fft PROPERTIES CUDA_STANDARD 11)
    target_compile_definitions(benchmark_fft PUBLIC ZISA_HAS_CUDA=1)
  endif()
  if(AZEBAN_HAS_MPI)
    target_compile_definitions(benchmark_fft PUBLIC AZEBAN_HAS_MPI=1)
    if(ZISA_HAS_CUDA AND AZEBAN_HAS_CUDA_AWARE_MPI)
      target_compile_definitions(benchmark_fft PUBLIC AZEBAN_HAS_CUDA_AWARE_MPI=1)
    endif()
    target_include_directories(benchmark_fft PUBLIC ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(benchmark_fft PUBLIC ${MPI_CXX_COMPILE_FLAGS})
    target_link_libraries(benchmark_fft PUBLIC ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
  endif()
  target_link_libraries(benchmark_fft PRIVATE fmt::fmt)
  target_link_libraries(benchmark_fft PRIVATE azeban_lib)
  target_link_libraries(benchmark_fft PRIVATE azeban_warning_flags)
endif()

# -- Postprocessing Standalone -------------------------------------------------
add_executable(postprocess)
target_link_libraries(postprocess PRIVATE azeban_lib)
target_link_libraries(postprocess PRIVATE azeban_warning_flags)
set_target_properties(postprocess PROPERTIES CXX_STANDARD 17)

# -- Subdirectories ------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(test)

if(AZEBAN_HAS_BENCHMARK)
  add_subdirectory(benchmarks)
endif()
