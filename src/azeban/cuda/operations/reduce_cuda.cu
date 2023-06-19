#include <azeban/cuda/operations/reduce_cuda.hpp>
#include <zisa/config.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/comparison.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

static __device__ void warpReduce(volatile real_t *sdata, unsigned tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void
reduce_sum_cuda_kernel(zisa::array_const_view<real_t, 1> in_data,
                       zisa::array_view<real_t, 1> out_data) {
  extern __shared__ real_t sdata[];

  zisa::int_t tid = threadIdx.x;
  zisa::int_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  if (i >= in_data.shape(0)) {
    sdata[tid] = 0;
  } else {
    sdata[tid] = in_data[i];
  }
  if (i + blockDim.x < in_data.shape(0)) {
    sdata[tid] += in_data[i + blockDim.x];
  }
  __syncthreads();

  for (zisa::int_t s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid < 32) {
    warpReduce(sdata, tid);
  }

  if (tid == 0) {
    out_data[blockIdx.x] = sdata[0];
  }
}

real_t reduce_sum_cuda(const zisa::array_const_view<real_t, 1> &data) {
  const int thread_dims = 1024;
  int block_dims = zisa::div_up(
      data.shape(0), zisa::integer_cast<zisa::int_t>(2 * thread_dims));
  auto out_data = zisa::cuda_array<real_t, 1>(zisa::shape_t<1>(block_dims));
  reduce_sum_cuda_kernel<<<block_dims,
                           thread_dims,
                           thread_dims * sizeof(real_t)>>>(
      data, zisa::array_view<real_t, 1>(out_data));
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
  while (block_dims > 1) {
    block_dims = zisa::div_up(block_dims, 2 * thread_dims);
    reduce_sum_cuda_kernel<<<block_dims,
                             thread_dims,
                             thread_dims * sizeof(real_t)>>>(
        zisa::array_const_view<real_t, 1>(out_data),
        zisa::array_view<real_t, 1>(out_data));
    cudaDeviceSynchronize();
    ZISA_CHECK_CUDA_DEBUG;
  }
  zisa::array<real_t, 1> value(zisa::shape_t<1>(1));
  zisa::internal::copy(
      value.raw(), value.device(), out_data.raw(), out_data.device(), 1);
  return value[0];
}

}
