#! /usr/bin/env bash

set -e

if [[ "$#" -lt 2 ]]
then
    echo "Usage: $0 COMPILER DESTINATION [--zisa_has_mpi=ZISA_HAS_MPI]"
fi

for arg in "$@"
do
    case $arg in
        --zisa_has_mpi=*)
            ZISA_HAS_MPI=${arg#*=}
            ;;
        *)
            ;;
    esac
done

if [[ -z "${ZISA_HAS_MPI}" ]]
then
    ZISA_HAS_MPI=0
fi

compiler="$1"
compiler_id="$(basename "${compiler}")"
compiler_version="$("$compiler" -dumpversion)"

build_sha="$(echo "${compiler}__${compiler_version}__ZISA_HAS_MPI=${ZISA_HAS_MPI}" | sha1sum | head -c 6)"

install_root="${PWD}/$2"
install_dir="${install_root}/${build_sha}"

echo ${install_dir}
