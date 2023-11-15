FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG ENABLE_PROFILING=OFF
ARG SINGLE_PRECISION=ON

WORKDIR azeban

COPY cmake cmake
COPY benchmarks benchmarks
COPY include include
COPY src src
COPY test test
COPY CMakeLists.txt .

RUN apt-get update && \
    apt-get install --no-install-recommends -y git libssl-dev catch2 libnetcdf-dev libpnetcdf-dev python3-dev pybind11-dev libopenmpi-dev fftw3-dev nlohmann-json3-dev libfmt-dev libboost-program-options-dev

RUN git clone https://github.com/Kitware/CMake.git --branch v3.28.0-rc3 && \
    cd CMake && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install

RUN git clone https://gitlab.kitware.com/paraview/catalyst.git --branch v2.0.0-rc4 && \
    cd catalyst && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCATALYST_BUILD_TESTING=OFF -B build && \
    cmake --build build -j$(nproc) && \
    cmake --install build

RUN git clone https://github.com/1uc/ZisaCore.git && \
    cmake -DCMAKE_BUILD_TYPE=Release -DZISA_HAS_CUDA=1 -S ZisaCore -B ZisaCore/build && \
    cmake --build ZisaCore/build -j$(nproc) && \
    cmake --install ZisaCore/build

RUN git clone https://github.com/1uc/ZisaMemory.git && \
    cmake -DCMAKE_BUILD_TYPE=Release -DZISA_HAS_CUDA=1 -DZISA_HAS_NETCDF=1 -DCMAKE_MODULE_PATH=cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/cmake/zisa -S ZisaMemory -B ZisaMemory/build && \
    cmake --build ZisaMemory/build -j$(nproc) && \
    cmake --install ZisaMemory/build

RUN cmake -DCMAKE_MODULE_PATH=cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/cmake/zisa -DCMAKE_BUILD_TYPE=Release -DENABLE_PROFILING=${ENABLE_PROFILING} -DSINGLE_PRECISION=${SINGLE_PRECISION} -DENABLE_PYTHON=ON -DENABLE_INSITU=ON -DENABLE_BENCHMARKS=OFF -S . -B build && \
    cmake --build build -j$(nproc)

ENTRYPOINT ["/azeban/build/azeban"]
