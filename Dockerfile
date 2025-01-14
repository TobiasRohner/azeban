FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG ENABLE_PROFILING=OFF
ARG SINGLE_PRECISION=ON

WORKDIR azeban

RUN apt-get update && \
    apt-get install --no-install-recommends -y git cmake wget libssl-dev libhdf5-mpi-dev python3-dev libfftw3-dev python3-numpy

RUN wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.0.tar.gz \
    && tar -xvf openmpi-5.0.0.tar.gz \
    && cd openmpi-5.0.0 \
    && ./configure --with-pmix \
    && make -j$(nproc) \
    && make install

RUN git clone https://gitlab.kitware.com/paraview/catalyst.git --branch v2.0.0-rc4 && \
    cd catalyst && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCATALYST_BUILD_TESTING=OFF -B build && \
    cmake --build build -j$(nproc) && \
    cmake --install build

COPY cmake cmake
COPY benchmarks benchmarks
COPY include include
COPY src src
COPY test test
COPY CMakeLists.txt .

RUN cmake -DCMAKE_MODULE_PATH=cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PROFILING=${ENABLE_PROFILING} -DSINGLE_PRECISION=${SINGLE_PRECISION} -DENABLE_PYTHON=ON -DENABLE_INSITU=ON -DENABLE_BENCHMARKS=OFF -S . -B build && \
    cmake --build build

ENTRYPOINT ["/azeban/build/azeban"]
