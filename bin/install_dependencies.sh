#! /usr/bin/env bash

set -e

if [[ "$#" -lt 2 ]]
then
    echo "Usage: $0 COMPILER DESTINATION [--zisa_has_cuda=ZISA_HAS_CUDA]"
    echo "                               [--cmake=CUSTOM_CMAKE_BINARY]"
    exit -1
fi

for arg in "$@"
do
    case $arg in
        --zisa_has_cuda=*)
            ZISA_HAS_CUDA=${arg#*=}
            ;;
        --cmake=*)
            CMAKE="$(realpath "${arg#*=}")"
            ;;
        *)
            ;;
    esac
done

if [[ -z "${CMAKE}" ]]
then
    CMAKE=cmake
fi


if [[ -z "${ZISA_HAS_CUDA}" ]]
then
    ZISA_HAS_CUDA=0
fi

component_name="azeban"
zisa_dependencies=("ZisaCore" "ZisaMemory")

azeban_root="$(realpath "$(dirname "$(readlink -f "$0")")"/..)"

CC="$1"
CXX="$(${azeban_root}/bin/cc2cxx.sh $CC)"
compiler_version=$("${CC}" -dumpversion)

install_dir="$("${azeban_root}/bin/install_dir.sh" "$1" "$2" --zisa_has_mpi=${ZISA_HAS_MPI})"
source_dir="${install_dir}/sources"
conan_file="${azeban_root}/conanfile.txt"

if [[ -f "$conan_file" ]]
then
   mkdir -p "${install_dir}/conan" && cd "${install_dir}/conan"
   conan install "$conan_file" \
         -s compiler=$(basename "${CC}") \
         -s compiler.libcxx=libstdc++11 \
	 --build=fmt
fi

mkdir -p "${source_dir}"
echo "The dependencies were installed at"
echo "    export DEP_DIR=${install_dir}"
echo ""
echo "Use"
echo "    ${CMAKE} \\ "
echo "        -DCMAKE_PROJECT_${component_name}_INCLUDE=${install_dir}/conan/conan_paths.cmake \\ "
echo "        -DCMAKE_MODULE_PATH=${install_dir}/conan \\ "
echo "        -DCMAKE_PREFIX_PATH=${install_dir}/zisa/lib/cmake/zisa \\ "
echo "        -DCMAKE_C_COMPILER=${CC} \\ "
echo "        -DCMAKE_CXX_COMPILER=${CXX} \\ "
echo "        -DZISA_HAS_CUDA=${ZISA_HAS_CUDA} \\ "
echo "        -DZISA_HAS_NETCDF=1 \\ "
echo "        REMAINING_ARGS "
