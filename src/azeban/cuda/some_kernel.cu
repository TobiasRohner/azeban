// This file only serves to make the build system happy. In particular prevent
// the complaint about no source files for target XYZ.

#include <zisa/config.hpp>
#include <zisa/cuda/cuda.hpp>

__global__ void some_kernel(double * x) {
  x[0] = 1.0;
}

