#include <azeban/cuda/operations/reduce_cuda.hpp>
#include <zisa/config.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/comparison.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template<typename Scalar>
static __device__ void warpReduce(volatile Scalar *sdata, unsigned tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void
reduce_sum_cuda_kernel_float(zisa::array_const_view<float, 1> in_data,
                       zisa::array_view<float, 1> out_data) {
  extern __shared__ float sdataf[];

  zisa::int_t tid = threadIdx.x;
  zisa::int_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  if (i >= in_data.shape(0)) {
    sdataf[tid] = 0;
  } else {
    sdataf[tid] = in_data[i];
  }
  if (i + blockDim.x < in_data.shape(0)) {
    sdataf[tid] += in_data[i + blockDim.x];
  }
  __syncthreads();

  for (zisa::int_t s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdataf[tid] += sdataf[tid + s];
    }
    __syncthreads();
  }
  if (tid < 32) {
    warpReduce(sdataf, tid);
  }

  if (tid == 0) {
    out_data[blockIdx.x] = sdataf[0];
  }
}

__global__ void
reduce_sum_cuda_kernel_double(zisa::array_const_view<double,  1> in_data,
                       zisa::array_view<double, 1> out_data) {
  extern __shared__ double sdatad[];

  zisa::int_t tid = threadIdx.x;
  zisa::int_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  if (i >= in_data.shape(0)) {
    sdatad[tid] = 0;
  } else {
    sdatad[tid] = in_data[i];
  }
  if (i + blockDim.x < in_data.shape(0)) {
    sdatad[tid] += in_data[i + blockDim.x];
  }
  __syncthreads();

  for (zisa::int_t s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdatad[tid] += sdatad[tid + s];
    }
    __syncthreads();
  }
  if (tid < 32) {
    warpReduce(sdatad, tid);
  }

  if (tid == 0) {
    out_data[blockIdx.x] = sdatad[0];
  }
}

float reduce_sum_cuda_float_impl(const zisa::array_const_view<float, 1> &data) {
  const int thread_dims = 1024;
  int block_dims = zisa::div_up(
      data.shape(0), zisa::integer_cast<zisa::int_t>(2 * thread_dims));
  auto out_data = zisa::cuda_array<float, 1>(zisa::shape_t<1>(block_dims));
  reduce_sum_cuda_kernel_float<<<block_dims,
                           thread_dims,
                           thread_dims * sizeof(float)>>>(
      data, zisa::array_view<float, 1>(out_data));
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
  while (block_dims > 1) {
    block_dims = zisa::div_up(block_dims, 2 * thread_dims);
    reduce_sum_cuda_kernel_float<<<block_dims,
                             thread_dims,
                             thread_dims * sizeof(float)>>>(
        zisa::array_const_view<float, 1>(out_data),
        zisa::array_view<float, 1>(out_data));
    cudaDeviceSynchronize();
    ZISA_CHECK_CUDA_DEBUG;
  }
  zisa::array<float, 1> value(zisa::shape_t<1>(1));
  zisa::internal::copy(
      value.raw(), value.device(), out_data.raw(), out_data.device(), 1);
  return value[0];
}

double reduce_sum_cuda_double_impl(const zisa::array_const_view<double, 1> &data) {
  const int thread_dims = 1024;
  int block_dims = zisa::div_up(
      data.shape(0), zisa::integer_cast<zisa::int_t>(2 * thread_dims));
  auto out_data = zisa::cuda_array<double, 1>(zisa::shape_t<1>(block_dims));
  reduce_sum_cuda_kernel_double<<<block_dims,
                           thread_dims,
                           thread_dims * sizeof(double)>>>(
      data, zisa::array_view<double, 1>(out_data));
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
  while (block_dims > 1) {
    block_dims = zisa::div_up(block_dims, 2 * thread_dims);
    reduce_sum_cuda_kernel_double<<<block_dims,
                             thread_dims,
                             thread_dims * sizeof(double)>>>(
        zisa::array_const_view<double, 1>(out_data),
        zisa::array_view<double, 1>(out_data));
    cudaDeviceSynchronize();
    ZISA_CHECK_CUDA_DEBUG;
  }
  zisa::array<double, 1> value(zisa::shape_t<1>(1));
  zisa::internal::copy(
      value.raw(), value.device(), out_data.raw(), out_data.device(), 1);
  return value[0];
}

float reduce_sum_cuda(const zisa::array_const_view<float, 1> &data) {
  return reduce_sum_cuda_float_impl(data);
}

double reduce_sum_cuda(const zisa::array_const_view<double, 1> &data) {
  return reduce_sum_cuda_double_impl(data);
}

}
