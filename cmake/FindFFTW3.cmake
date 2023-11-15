# - Find the FFTW3 library
#
# Original version of this file:
#   Copyright (c) 2015, Wenzel Jakob
#   https://github.com/wjakob/layerlab/blob/master/cmake/FindFFTW3.cmake, commit 4d58bfdc28891b4f9373dfe46239dda5a0b561c6
# Modifications:
#   Copyright (c) 2017, Patrick Bos
#
# Usage:
#   find_package(FFTW3 [REQUIRED] [QUIET] [COMPONENTS component1 ... componentX] )
#
# It sets the following variables:
#   FFTW3_FOUND                  ... true if fftw is found on the system
#   FFTW3_[component]_LIB_FOUND  ... true if the component is found on the system (see components below)
#   FFTW3_LIBRARIES              ... full paths to all found fftw libraries
#   FFTW3_[component]_LIB        ... full path to one of the components (see below)
#   FFTW3_INCLUDE_DIRS           ... fftw include directory paths
#
# The following variables will be checked by the function
#   FFTW3_USE_STATIC_LIBS        ... if true, only static libraries are found, otherwise both static and shared.
#   FFTW3_ROOT                   ... if set, the libraries are exclusively searched
#                                   under this path
#
# This package supports the following components:
#   FLOAT_LIB
#   DOUBLE_LIB
#   LONGDOUBLE_LIB
#   FLOAT_THREADS_LIB
#   DOUBLE_THREADS_LIB
#   LONGDOUBLE_THREADS_LIB
#   FLOAT_OPENMP_LIB
#   DOUBLE_OPENMP_LIB
#   LONGDOUBLE_OPENMP_LIB
#

# TODO (maybe): extend with ExternalProject download + build option
# TODO: put on conda-forge


if( NOT FFTW3_ROOT AND DEFINED ENV{FFTW3DIR} )
    set( FFTW3_ROOT $ENV{FFTW3DIR} )
endif()

# Check if we can use PkgConfig
find_package(PkgConfig)

#Determine from PKG
if( PKG_CONFIG_FOUND AND NOT FFTW3_ROOT )
    pkg_check_modules( PKG_FFTW3 QUIET "fftw3" )
endif()

#Check whether to search static or dynamic libs
set( CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES} )

if( ${FFTW3_USE_STATIC_LIBS} )
    set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
else()
    set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV} )
endif()

if( FFTW3_ROOT )
    # find libs

    find_library(
        FFTW3_DOUBLE_LIB
        NAMES "fftw3" libfftw3-3
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    find_library(
        FFTW3_DOUBLE_THREADS_LIB
        NAMES "fftw3_threads"
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    find_library(
        FFTW3_DOUBLE_OPENMP_LIB
        NAMES "fftw3_omp"
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    find_library(
        FFTW3_DOUBLE_MPI_LIB
        NAMES "fftw3_mpi"
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    find_library(
        FFTW3_FLOAT_LIB
        NAMES "fftw3f" libfftw3f-3
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    find_library(
        FFTW3_FLOAT_THREADS_LIB
        NAMES "fftw3f_threads"
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    find_library(
        FFTW3_FLOAT_OPENMP_LIB
        NAMES "fftw3f_omp"
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    find_library(
        FFTW3_FLOAT_MPI_LIB
        NAMES "fftw3f_mpi"
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    find_library(
        FFTW3_LONGDOUBLE_LIB
        NAMES "fftw3l" libfftw3l-3
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    find_library(
        FFTW3_LONGDOUBLE_THREADS_LIB
        NAMES "fftw3l_threads"
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    find_library(
        FFTW3_LONGDOUBLE_OPENMP_LIB
        NAMES "fftw3l_omp"
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    find_library(
        FFTW3_LONGDOUBLE_MPI_LIB
        NAMES "fftw3l_mpi"
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "lib" "lib64"
        NO_DEFAULT_PATH
        )

    #find includes
    find_path(FFTW3_INCLUDE_DIRS
        NAMES "fftw3.h"
        PATHS ${FFTW3_ROOT}
        PATH_SUFFIXES "include"
        NO_DEFAULT_PATH
        )

else()

    find_library(
        FFTW3_DOUBLE_LIB
        NAMES "fftw3"
        PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )

    find_library(
        FFTW3_DOUBLE_THREADS_LIB
        NAMES "fftw3_threads"
        PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )

    find_library(
        FFTW3_DOUBLE_OPENMP_LIB
        NAMES "fftw3_omp"
        PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )

    find_library(
        FFTW3_DOUBLE_MPI_LIB
        NAMES "fftw3_mpi"
        PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )

    find_library(
        FFTW3_FLOAT_LIB
        NAMES "fftw3f"
        PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )

    find_library(
        FFTW3_FLOAT_THREADS_LIB
        NAMES "fftw3f_threads"
        PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )

    find_library(
        FFTW3_FLOAT_OPENMP_LIB
        NAMES "fftw3f_omp"
        PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )

    find_library(
        FFTW3_FLOAT_MPI_LIB
        NAMES "fftw3f_mpi"
        PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )

    find_library(
        FFTW3_LONGDOUBLE_LIB
        NAMES "fftw3l"
        PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )

    find_library(
        FFTW3_LONGDOUBLE_THREADS_LIB
        NAMES "fftw3l_threads"
        PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )

    find_library(FFTW3_LONGDOUBLE_OPENMP_LIB
        NAMES "fftw3l_omp"
        PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )

    find_library(FFTW3_LONGDOUBLE_MPI_LIB
        NAMES "fftw3l_mpi"
        PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
        )

    find_path(FFTW3_INCLUDE_DIRS
        NAMES "fftw3.h"
        PATHS ${PKG_FFTW3_INCLUDE_DIRS} ${INCLUDE_INSTALL_DIR}
        )

endif( FFTW3_ROOT )

#--------------------------------------- components

if (FFTW3_DOUBLE_LIB)
    set(FFTW3_DOUBLE_LIB_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_DOUBLE_LIB})
    add_library(FFTW3::Double INTERFACE IMPORTED)
    set_target_properties(FFTW3::Double
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_DOUBLE_LIB}"
        )
else()
    set(FFTW3_DOUBLE_LIB_FOUND FALSE)
endif()

if (FFTW3_FLOAT_LIB)
    set(FFTW3_FLOAT_LIB_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_FLOAT_LIB})
    add_library(FFTW3::Float INTERFACE IMPORTED)
    set_target_properties(FFTW3::Float
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_FLOAT_LIB}"
        )
else()
    set(FFTW3_FLOAT_LIB_FOUND FALSE)
endif()

if (FFTW3_LONGDOUBLE_LIB)
    set(FFTW3_LONGDOUBLE_LIB_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_LONGDOUBLE_LIB})
    add_library(FFTW3::LongDouble INTERFACE IMPORTED)
    set_target_properties(FFTW3::LongDouble
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_LONGDOUBLE_LIB}"
        )
else()
    set(FFTW3_LONGDOUBLE_LIB_FOUND FALSE)
endif()

if (FFTW3_DOUBLE_THREADS_LIB)
    set(FFTW3_DOUBLE_THREADS_LIB_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_DOUBLE_THREADS_LIB})
    add_library(FFTW3::DoubleThreads INTERFACE IMPORTED)
    set_target_properties(FFTW3::DoubleThreads
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_DOUBLE_THREADS_LIB}"
        )
else()
    set(FFTW3_DOUBLE_THREADS_LIB_FOUND FALSE)
endif()

if (FFTW3_FLOAT_THREADS_LIB)
    set(FFTW3_FLOAT_THREADS_LIB_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_FLOAT_THREADS_LIB})
    add_library(FFTW3::FloatThreads INTERFACE IMPORTED)
    set_target_properties(FFTW3::FloatThreads
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_FLOAT_THREADS_LIB}"
        )
else()
    set(FFTW3_FLOAT_THREADS_LIB_FOUND FALSE)
endif()

if (FFTW3_LONGDOUBLE_THREADS_LIB)
    set(FFTW3_LONGDOUBLE_THREADS_LIB_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_LONGDOUBLE_THREADS_LIB})
    add_library(FFTW3::LongDoubleThreads INTERFACE IMPORTED)
    set_target_properties(FFTW3::LongDoubleThreads
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_LONGDOUBLE_THREADS_LIB}"
        )
else()
    set(FFTW3_LONGDOUBLE_THREADS_LIB_FOUND FALSE)
endif()

if (FFTW3_DOUBLE_OPENMP_LIB)
    set(FFTW3_DOUBLE_OPENMP_LIB_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_DOUBLE_OPENMP_LIB})
    add_library(FFTW3::DoubleOpenMP INTERFACE IMPORTED)
    set_target_properties(FFTW3::DoubleOpenMP
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_DOUBLE_OPENMP_LIB}"
        )
else()
    set(FFTW3_DOUBLE_OPENMP_LIB_FOUND FALSE)
endif()

if (FFTW3_FLOAT_OPENMP_LIB)
    set(FFTW3_FLOAT_OPENMP_LIB_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_FLOAT_OPENMP_LIB})
    add_library(FFTW3::FloatOpenMP INTERFACE IMPORTED)
    set_target_properties(FFTW3::FloatOpenMP
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_FLOAT_OPENMP_LIB}"
        )
else()
    set(FFTW3_FLOAT_OPENMP_LIB_FOUND FALSE)
endif()

if (FFTW3_LONGDOUBLE_OPENMP_LIB)
    set(FFTW3_LONGDOUBLE_OPENMP_LIB_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_LONGDOUBLE_OPENMP_LIB})
    add_library(FFTW3::LongDoubleOpenMP INTERFACE IMPORTED)
    set_target_properties(FFTW3::LongDoubleOpenMP
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_LONGDOUBLE_OPENMP_LIB}"
        )
else()
    set(FFTW3_LONGDOUBLE_OPENMP_LIB_FOUND FALSE)
endif()

if (FFTW3_DOUBLE_MPI_LIB)
    set(FFTW3_DOUBLE_MPI_LIB_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_DOUBLE_MPI_LIB})
    add_library(FFTW3::DoubleMPI INTERFACE IMPORTED)
    set_target_properties(FFTW3::DoubleMPI
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_DOUBLE_MPI_LIB}"
        )
else()
    set(FFTW3_DOUBLE_MPI_LIB_FOUND FALSE)
endif()

if (FFTW3_FLOAT_MPI_LIB)
    set(FFTW3_FLOAT_MPI_LIB_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_FLOAT_MPI_LIB})
    add_library(FFTW3::FloatMPI INTERFACE IMPORTED)
    set_target_properties(FFTW3::FloatMPI
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_FLOAT_MPI_LIB}"
        )
else()
    set(FFTW3_FLOAT_MPI_LIB_FOUND FALSE)
endif()

if (FFTW3_LONGDOUBLE_MPI_LIB)
    set(FFTW3_LONGDOUBLE_MPI_LIB_FOUND TRUE)
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_LONGDOUBLE_MPI_LIB})
    add_library(FFTW3::LongDoubleMPI INTERFACE IMPORTED)
    set_target_properties(FFTW3::LongDoubleMPI
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${FFTW3_LONGDOUBLE_MPI_LIB}"
        )
else()
    set(FFTW3_LONGDOUBLE_MPI_LIB_FOUND FALSE)
endif()

#--------------------------------------- end components

set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV} )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(FFTW3
    REQUIRED_VARS FFTW3_INCLUDE_DIRS
    HANDLE_COMPONENTS
    )

mark_as_advanced(
    FFTW3_INCLUDE_DIRS
    FFTW3_LIBRARIES
    FFTW3_FLOAT_LIB
    FFTW3_DOUBLE_LIB
    FFTW3_LONGDOUBLE_LIB
    FFTW3_FLOAT_THREADS_LIB
    FFTW3_DOUBLE_THREADS_LIB
    FFTW3_LONGDOUBLE_THREADS_LIB
    FFTW3_FLOAT_OPENMP_LIB
    FFTW3_DOUBLE_OPENMP_LIB
    FFTW3_LONGDOUBLE_OPENMP_LIB
    FFTW3_FLOAT_MPI_LIB
    FFTW3_DOUBLE_MPI_LIB
    FFTW3_LONGDOUBLE_MPI_LIB
    )
