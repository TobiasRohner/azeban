/*
 * This file is part of azeban (https://github.com/TobiasRohner/azeban).
 * Copyright (c) 2021 Tobias Rohner.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef AZEBAN_CUDA_OPERATIONS_NORM_CUDA_IMPL_HPP_
#define AZEBAN_CUDA_OPERATIONS_NORM_CUDA_IMPL_HPP_

#include "norm_cuda.hpp"
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

static __device__ void warpReduceMax(volatile real_t *sdata, unsigned tid) {
  sdata[tid] = zisa::max(sdata[tid], sdata[tid + 32]);
  sdata[tid] = zisa::max(sdata[tid], sdata[tid + 16]);
  sdata[tid] = zisa::max(sdata[tid], sdata[tid + 8]);
  sdata[tid] = zisa::max(sdata[tid], sdata[tid + 4]);
  sdata[tid] = zisa::max(sdata[tid], sdata[tid + 2]);
  sdata[tid] = zisa::max(sdata[tid], sdata[tid + 1]);
}

template <typename Scalar>
__global__ void norm_cuda_kernel(zisa::array_const_view<Scalar, 1> in_data,
                                 zisa::array_view<real_t, 1> out_data,
                                 real_t p) {
  extern __shared__ real_t sdata[];

  zisa::int_t tid = threadIdx.x;
  zisa::int_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
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
    out_data[blockIdx.x] = zisa::pow(sdata[0], real_t(1. / p));
  }
}

template <typename Scalar>
__global__ void max_norm_cuda_kernel(zisa::array_const_view<Scalar, 1> in_data,
                                     zisa::array_view<real_t, 1> out_data) {
  extern __shared__ real_t sdata[];

  zisa::int_t tid = threadIdx.x;
  zisa::int_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  using zisa::abs;
  if (i >= in_data.shape(0)) {
    sdata[tid] = 0;
  } else {
    sdata[tid] = abs(in_data[i]);
  }
  if (i + blockDim.x < in_data.shape(0)) {
    sdata[tid] = zisa::max(sdata[tid], real_t(abs(in_data[i + blockDim.x])));
  }
  __syncthreads();

  for (zisa::int_t s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] = zisa::max(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
  if (tid < 32) {
    warpReduceMax(sdata, tid);
  }

  if (tid == 0) {
    out_data[blockIdx.x] = sdata[0];
  }
}

template <typename Scalar>
real_t norm_cuda(const zisa::array_const_view<Scalar, 1> &data, real_t p) {
  const int thread_dims = 1024;
  int block_dims = zisa::div_up(
      data.shape(0), zisa::integer_cast<zisa::int_t>(2 * thread_dims));
  auto out_data = zisa::cuda_array<real_t, 1>(zisa::shape_t<1>(block_dims));
  norm_cuda_kernel<<<block_dims, thread_dims, thread_dims * sizeof(real_t)>>>(
      data, zisa::array_view<real_t, 1>(out_data), p);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
  while (block_dims > 1) {
    block_dims = zisa::div_up(block_dims, 2 * thread_dims);
    norm_cuda_kernel<<<block_dims, thread_dims, thread_dims * sizeof(real_t)>>>(
        zisa::array_const_view<real_t, 1>(out_data),
        zisa::array_view<real_t, 1>(out_data),
        p);
    cudaDeviceSynchronize();
    ZISA_CHECK_CUDA_DEBUG;
  }
  zisa::array<real_t, 1> value(zisa::shape_t<1>(1));
  zisa::internal::copy(
      value.raw(), value.device(), out_data.raw(), out_data.device(), 1);
  return value[0];
}

template <typename Scalar>
real_t max_norm_cuda(const zisa::array_const_view<Scalar, 1> &data) {
  const int thread_dims = 1024;
  int block_dims = zisa::div_up(
      data.shape(0), zisa::integer_cast<zisa::int_t>(2 * thread_dims));
  auto out_data = zisa::cuda_array<real_t, 1>(zisa::shape_t<1>(block_dims));
  max_norm_cuda_kernel<<<block_dims,
                         thread_dims,
                         thread_dims * sizeof(real_t)>>>(
      data, zisa::array_view<real_t, 1>(out_data));
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
  while (block_dims > 1) {
    block_dims = zisa::div_up(block_dims, 2 * thread_dims);
    max_norm_cuda_kernel<<<block_dims,
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

#endif
