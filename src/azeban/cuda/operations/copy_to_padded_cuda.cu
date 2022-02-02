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
#include <azeban/config.hpp>
#include <azeban/cuda/operations/copy_to_padded_cuda.hpp>
#include <azeban/operations/copy_to_padded_functions.hpp>
#include <zisa/config.hpp>

namespace azeban {

template <bool pad_x, int compact_dim>
__global__ void
copy_to_padded_cuda_kernel(zisa::array_view<complex_t, 1> dst,
                           zisa::array_const_view<complex_t, 1> src,
                           complex_t pad_value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dst.shape(0)) {
    return;
  }

  const auto src_shape = src.shape();
  const auto dst_shape = dst.shape();
  const int idx_dst = zisa::row_major<1>::linear_index(dst_shape, i);
  int i_src = i;
  if (pad_dim<pad_x, compact_dim == 0>(
          dst, idx_dst, src_shape[0], dst_shape[0], i, pad_value, &i_src)) {
    return;
  }
  const int idx_src = zisa::row_major<1>::linear_index(src_shape, i_src);
  dst[idx_dst] = src[idx_src];
}

template <bool pad_x, bool pad_y, int compact_dim>
__global__ void
copy_to_padded_cuda_kernel(zisa::array_view<complex_t, 2> dst,
                           zisa::array_const_view<complex_t, 2> src,
                           complex_t pad_value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const auto src_shape = src.shape();
  const auto dst_shape = dst.shape();
  if (i >= dst_shape[0]) {
    return;
  }
  if (j >= dst_shape[1]) {
    return;
  }

  const int idx_dst = zisa::row_major<2>::linear_index(dst_shape, i, j);
  int i_src = i;
  int j_src = j;
  if (pad_dim<pad_x, compact_dim == 0>(
          dst, idx_dst, src_shape[0], dst_shape[0], i, pad_value, &i_src)) {
    return;
  }
  if (pad_dim<pad_y, compact_dim == 1>(
          dst, idx_dst, src_shape[1], dst_shape[1], j, pad_value, &j_src)) {
    return;
  }
  const int idx_src = zisa::row_major<2>::linear_index(src_shape, i_src, j_src);
  dst[idx_dst] = src[idx_src];
}

template <bool pad_x, bool pad_y, bool pad_z, int compact_dim>
__global__ void
copy_to_padded_cuda_kernel(zisa::array_view<complex_t, 3> dst,
                           zisa::array_const_view<complex_t, 3> src,
                           complex_t pad_value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;
  const auto src_shape = src.shape();
  const auto dst_shape = dst.shape();
  if (i >= dst_shape[0]) {
    return;
  }
  if (j >= dst_shape[1]) {
    return;
  }
  if (k >= dst_shape[2]) {
    return;
  }

  const int idx_dst = zisa::row_major<3>::linear_index(dst_shape, i, j, k);
  int i_src = i;
  int j_src = j;
  int k_src = k;
  if (pad_dim<pad_x, compact_dim == 0>(
          dst, idx_dst, src_shape[0], dst_shape[0], i, pad_value, &i_src)) {
    return;
  }
  if (pad_dim<pad_y, compact_dim == 1>(
          dst, idx_dst, src_shape[1], dst_shape[1], j, pad_value, &j_src)) {
    return;
  }
  if (pad_dim<pad_z, compact_dim == 2>(
          dst, idx_dst, src_shape[2], dst_shape[2], k, pad_value, &k_src)) {
    return;
  }
  const int idx_src
      = zisa::row_major<3>::linear_index(src_shape, i_src, j_src, k_src);
  dst[idx_dst] = src[idx_src];
}

template <bool pad_x, int compact_dim>
void copy_to_padded_cuda(const zisa::array_view<complex_t, 1> &dst,
                         const zisa::array_const_view<complex_t, 1> &src,
                         const complex_t &pad_value) {
  assert(src.memory_location() == zisa::device_type::cuda);
  assert(dst.memory_location() == zisa::device_type::cuda);
  assert(dst.shape(0) >= src.shape(0));
  const int thread_dims = 1024;
  const int block_dims = zisa::min(
      zisa::div_up(static_cast<int>(dst.shape(0)), thread_dims), 1024);
  copy_to_padded_cuda_kernel<pad_x, compact_dim>
      <<<block_dims, thread_dims>>>(dst, src, pad_value);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <int compact_dim>
void copy_to_padded_cuda(bool pad_x,
                         const zisa::array_view<complex_t, 1> &dst,
                         const zisa::array_const_view<complex_t, 1> &src,
                         const complex_t &pad_value) {
  if (pad_x) {
    copy_to_padded_cuda<true, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x) {
    copy_to_padded_cuda<false, compact_dim>(dst, src, pad_value);
  }
}

void copy_to_padded_cuda(bool pad_x,
                         int compact_dim,
                         const zisa::array_view<complex_t, 1> &dst,
                         const zisa::array_const_view<complex_t, 1> &src,
                         const complex_t &pad_value) {
  switch (compact_dim) {
  case 0:
    copy_to_padded_cuda<0>(pad_x, dst, src, pad_value);
    return;
  default:
    copy_to_padded_cuda<-1>(pad_x, dst, src, pad_value);
    return;
  }
}

void copy_to_padded_cuda(const zisa::array_view<complex_t, 1> &dst,
                         const zisa::array_const_view<complex_t, 1> &src,
                         const complex_t &pad_value) {
  copy_to_padded_cuda<true, 0>(dst, src, pad_value);
}

template <bool pad_x, bool pad_y, int compact_dim>
void copy_to_padded_cuda(const zisa::array_view<complex_t, 2> &dst,
                         const zisa::array_const_view<complex_t, 2> &src,
                         const complex_t &pad_value) {
  assert(src.memory_location() == zisa::device_type::cuda);
  assert(dst.memory_location() == zisa::device_type::cuda);
  assert(dst.shape(0) >= src.shape(0));
  assert(dst.shape(1) >= src.shape(1));
  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::min(zisa::div_up(static_cast<int>(dst.shape(0)), thread_dims.x),
                1024),
      zisa::min(zisa::div_up(static_cast<int>(dst.shape(1)), thread_dims.y),
                1024),
      1);
  copy_to_padded_cuda_kernel<pad_x, pad_y, compact_dim>
      <<<block_dims, thread_dims>>>(dst, src, pad_value);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <int compact_dim>
void copy_to_padded_cuda(bool pad_x,
                         bool pad_y,
                         const zisa::array_view<complex_t, 2> &dst,
                         const zisa::array_const_view<complex_t, 2> &src,
                         const complex_t &pad_value) {
  if (pad_x && pad_y) {
    copy_to_padded_cuda<true, true, compact_dim>(dst, src, pad_value);
  }
  if (pad_x && !pad_y) {
    copy_to_padded_cuda<true, false, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x && pad_y) {
    copy_to_padded_cuda<false, true, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x && !pad_y) {
    copy_to_padded_cuda<false, false, compact_dim>(dst, src, pad_value);
  }
}

void copy_to_padded_cuda(bool pad_x,
                         bool pad_y,
                         int compact_dim,
                         const zisa::array_view<complex_t, 2> &dst,
                         const zisa::array_const_view<complex_t, 2> &src,
                         const complex_t &pad_value) {
  switch (compact_dim) {
  case 0:
    copy_to_padded_cuda<0>(pad_x, pad_y, dst, src, pad_value);
    return;
  case 1:
    copy_to_padded_cuda<1>(pad_x, pad_y, dst, src, pad_value);
    return;
  default:
    copy_to_padded_cuda<-1>(pad_x, pad_y, dst, src, pad_value);
    return;
  }
}

void copy_to_padded_cuda(const zisa::array_view<complex_t, 2> &dst,
                         const zisa::array_const_view<complex_t, 2> &src,
                         const complex_t &pad_value) {
  copy_to_padded_cuda<true, true, 1>(dst, src, pad_value);
}

template <bool pad_x, bool pad_y, bool pad_z, int compact_dim>
void copy_to_padded_cuda(const zisa::array_view<complex_t, 3> &dst,
                         const zisa::array_const_view<complex_t, 3> &src,
                         const complex_t &pad_value) {
  assert(src.memory_location() == zisa::device_type::cuda);
  assert(dst.memory_location() == zisa::device_type::cuda);
  assert(dst.shape(0) >= src.shape(0));
  assert(dst.shape(1) >= src.shape(1));
  assert(dst.shape(2) >= src.shape(2));
  const dim3 thread_dims(4, 4, 32);
  const dim3 block_dims(
      zisa::min(zisa::div_up(static_cast<int>(dst.shape(0)), thread_dims.x),
                1024),
      zisa::min(zisa::div_up(static_cast<int>(dst.shape(1)), thread_dims.y),
                1024),
      zisa::min(zisa::div_up(static_cast<int>(dst.shape(2)), thread_dims.z),
                1024));
  copy_to_padded_cuda_kernel<pad_x, pad_y, pad_z, compact_dim>
      <<<block_dims, thread_dims>>>(dst, src, pad_value);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <int compact_dim>
void copy_to_padded_cuda(bool pad_x,
                         bool pad_y,
                         bool pad_z,
                         const zisa::array_view<complex_t, 3> &dst,
                         const zisa::array_const_view<complex_t, 3> &src,
                         const complex_t &pad_value) {
  if (pad_x && pad_y && pad_z) {
    copy_to_padded_cuda<true, true, true, compact_dim>(dst, src, pad_value);
  }
  if (pad_x && pad_y && !pad_z) {
    copy_to_padded_cuda<true, true, false, compact_dim>(dst, src, pad_value);
  }
  if (pad_x && !pad_y && pad_z) {
    copy_to_padded_cuda<true, false, true, compact_dim>(dst, src, pad_value);
  }
  if (pad_x && !pad_y && !pad_z) {
    copy_to_padded_cuda<true, false, false, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x && pad_y && pad_z) {
    copy_to_padded_cuda<false, true, true, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x && pad_y && !pad_z) {
    copy_to_padded_cuda<false, true, false, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x && !pad_y && pad_z) {
    copy_to_padded_cuda<false, false, true, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x && !pad_y && !pad_z) {
    copy_to_padded_cuda<false, false, false, compact_dim>(dst, src, pad_value);
  }
}

void copy_to_padded_cuda(bool pad_x,
                         bool pad_y,
                         bool pad_z,
                         int compact_dim,
                         const zisa::array_view<complex_t, 3> &dst,
                         const zisa::array_const_view<complex_t, 3> &src,
                         const complex_t &pad_value) {
  switch (compact_dim) {
  case 0:
    copy_to_padded_cuda<0>(pad_x, pad_y, pad_z, dst, src, pad_value);
    return;
  case 1:
    copy_to_padded_cuda<1>(pad_x, pad_y, pad_z, dst, src, pad_value);
    return;
  case 2:
    copy_to_padded_cuda<2>(pad_x, pad_y, pad_z, dst, src, pad_value);
    return;
  default:
    copy_to_padded_cuda<-1>(pad_x, pad_y, pad_z, dst, src, pad_value);
    return;
  }
}

void copy_to_padded_cuda(const zisa::array_view<complex_t, 3> &dst,
                         const zisa::array_const_view<complex_t, 3> &src,
                         const complex_t &pad_value) {
  copy_to_padded_cuda<true, true, true, 2>(dst, src, pad_value);
}

}
