#!/usr/bin/env bash

set -e

if [[ $# -ne 1 ]]; then
  echo "Usage:"
  echo "    ${0} INSTALL_PREFIX"
  exit -1
fi

INSTALL_PREFIX=$(realpath $1)

CMAKE_MAJOR_VERSION=3
CMAKE_MINOR_VERSION=19
CMAKE_RELEASE_VERSION=4

TMP_DIR=$(mktemp -d --tmpdir=${SCRATCH})
CMAKE_DIR="cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_RELEASE_VERSION}"

cd $TMP_DIR
wget https://cmake.org/files/v${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}/cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_RELEASE_VERSION}.tar.gz
tar xvf cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_RELEASE_VERSION}.tar.gz

cd ${CMAKE_DIR}
./bootstrap --system-curl --prefix=$INSTALL_PREFIX
make -j$(nproc)
make install

cd ${HOME}
rm -rf $TMP_DIR
