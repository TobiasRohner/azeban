# -- Flags ---------------------------------------------------------------------
macro(host_compiler_flags)
  target_compile_options(azeban_warning_flags
    INTERFACE
    $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${ARGV}>>
    $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${ARGV}>>
    )
endmacro()

macro(nvcc_compiler_flags)
  target_compile_options(azeban_warning_flags
    INTERFACE
    $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:${ARGV}>
    )
endmacro()

add_library(azeban_warning_flags INTERFACE)
if(NOT TARGET warning_flags)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    host_compiler_flags(-Wall)
    host_compiler_flags(-Wextra)
    # host_compiler_flags(-Wconversion)
  endif()

  if(ZISA_MAX_ERRORS)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      host_compiler_flags(-fmax-errors=${ZISA_MAX_ERRORS})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      assert(${ZISA_HAS_CUDA} == 0)
      target_compile_options(azeban_warning_flags INTERFACE -ferror-limit=${ZISA_MAX_ERRORS})
    endif()
  endif()

  if(ZISA_HAS_CUDA)
    nvcc_compiler_flags(-Werror cross-execution-space-call)
  endif()
else()
  target_link_libraries(azeban_warning_flags INTERFACE warning_flags)
endif()
