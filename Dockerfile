FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

ARG ENABLE_PROFILING=OFF
ARG SINGLE_PRECISION=ON

WORKDIR azeban

RUN apt-get update && \
    #apt-get purge -y libpmix2 libpmix-dev libpmix* || true && \
    #apt-get autoremove -y && \
    apt-get install --no-install-recommends -y git cmake wget libssl-dev libhdf5-mpi-dev python3-dev libfftw3-dev python3-numpy vim gdb && \
    apt-get clean

ENV PATH=/usr/local/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

RUN wget https://github.com/openucx/ucx/releases/download/v1.18.1/ucx-1.18.1.tar.gz \
    && tar -xvf ucx-1.18.1.tar.gz \
    && cd ucx-1.18.1 \
    && ./configure CFLAGS="-fPIC" CXXFLAGS="-fPIC" CC="gcc -fPIC" CXX="g++ -fPIC" --with-cuda=/usr/local/cuda --enable-mt \
    && make -j$(nproc) \
    && make install \
    && ldconfig

RUN wget https://github.com/openpmix/openpmix/releases/download/v5.0.8/pmix-5.0.8.tar.gz \
    && tar -xvf pmix-5.0.8.tar.gz \
    && cd pmix-5.0.8 \
    && ./configure CFLAGS="-fPIC" CXXFLAGS="-fPIC" CC="gcc -fPIC" CXX="g++ -fPIC" --enable-static --disable-shared \
    && make -j$(nproc) \
    && make install \
    && ldconfig

RUN wget https://github.com/openpmix/prrte/releases/download/v3.0.11/prrte-3.0.11.tar.gz \
    && tar -xvf prrte-3.0.11.tar.gz \
    && cd prrte-3.0.11 \
    && ./configure CFLAGS="-fPIC" CXXFLAGS="-fPIC" CC="gcc -fPIC" CXX="g++ -fPIC" --with-pmix=/usr/local --enable-static --disable-shared --verbose \
    && make -j$(nproc) \
    && make install \
    && ldconfig

RUN wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.8.tar.gz \
    && tar -xvf openmpi-5.0.8.tar.gz \
    && cd openmpi-5.0.8 \
    && ./configure CFLAGS="-fPIC" CXXFLAGS="-fPIC" CC="gcc -fPIC" CXX="g++ -fPIC" --with-pmix=/usr/local --with-prte=/usr/local --with-cuda=/usr/local/cuda --with-ucx \
    && make -j$(nproc) \
    && make install \
    && ldconfig

RUN git clone https://gitlab.kitware.com/paraview/catalyst.git --branch v2.0.0 && \
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
    cmake --build build -j$(nproc)

ENTRYPOINT ["/azeban/build/azeban"]
