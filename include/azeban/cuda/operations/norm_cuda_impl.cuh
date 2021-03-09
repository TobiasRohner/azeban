#ifndef NNORM_CUDA_IMPL_H_
#define NNORM_CUDA_IMPL_H_

#include "norm_cuda.hpp"
#include <zisa/config.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/comparison.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

__device__ void warpReduce(volatile real_t *sdata, unsigned tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

template <typename Scalar>
__global__ void norm_cuda_kernel(zisa::array_const_view<Scalar, 1> in_data,
                                 zisa::array_view<real_t, 1> out_data,
                                 real_t p) {
  extern __shared__ real_t sdata[];

  unsigned tid = threadIdx.x;
  unsigned i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  using zisa::abs;
  if (i >= in_data.shape(0)) {
    sdata[tid] = 0;
  } else {
    sdata[tid] = zisa::pow(abs(in_data[i]), p);
  }
  if (i + blockDim.x < in_data.shape(0)) {
    sdata[tid] += zisa::pow(abs(in_data[i + blockDim.x]), p);
  }
  __syncthreads();

  for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid < 32) {
    warpReduce(sdata, tid);
  }

  if (tid == 0) {
    out_data[blockIdx.x] = zisa::pow(sdata[0], real_t(1. / p));
  }
}

template <typename Scalar>
real_t norm_cuda(const zisa::array_const_view<Scalar, 1> &data, real_t p) {
  const int thread_dims = 1024;
  int block_dims
      = zisa::div_up(static_cast<int>(data.shape(0)), 2 * thread_dims);
  auto out_data = zisa::cuda_array<real_t, 1>(zisa::shape_t<1>(block_dims));
  norm_cuda_kernel<<<block_dims, thread_dims, thread_dims * sizeof(real_t)>>>(
      data, zisa::array_view<real_t, 1>(out_data), p);
  while (block_dims > 1) {
    block_dims = zisa::div_up(static_cast<int>(block_dims), 2 * thread_dims);
    norm_cuda_kernel<<<block_dims, thread_dims, thread_dims * sizeof(real_t)>>>(
        zisa::array_const_view<real_t, 1>(out_data),
        zisa::array_view<real_t, 1>(out_data),
        p);
  }
  zisa::array<real_t, 1> value(zisa::shape_t<1>(1));
  zisa::internal::copy(
      value.raw(), value.device(), out_data.raw(), out_data.device(), 1);
  return value[0];
}

}

#endif
