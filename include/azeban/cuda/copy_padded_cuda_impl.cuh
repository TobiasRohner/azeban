#ifndef COPY_PADDED_CUDA_IMPL_H_
#define COPY_PADDED_CUDA_IMPL_H_

#include <zisa/memory/array_view.hpp>
#include <zisa/math/comparison.hpp>
#include <zisa/math/basic_functions.hpp>


namespace azeban {


template<typename T>
__global__ void copy_to_padded_cuda_kernel(zisa::array_view<T, 1> dst, zisa::array_const_view<T, 1> src, T pad_value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < src.shape(0)) {
    dst[idx] = src[idx];
  }
  else if (idx < dst.shape(0)) {
    dst[idx] = pad_value;
  }
}


template<typename T>
__global__ void copy_to_padded_cuda_kernel(zisa::array_view<T, 2> dst, zisa::array_const_view<T, 2> src, T pad_value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const auto src_shape = src.shape();
  const auto dst_shape = dst.shape();
  const int idx_src = zisa::row_major<2>::linear_index(src_shape, i, j);
  const int idx_dst = zisa::row_major<2>::linear_index(dst_shape, i, j);
  if (i < src_shape[0] && j < src_shape[1]) {
    dst[idx_dst] = src[idx_src];
  }
  else if (i < dst_shape[0] && j < dst_shape[1]) {
    dst[idx_dst] = pad_value;
  }
}


template<typename T>
__global__ void copy_to_padded_cuda_kernel(zisa::array_view<T, 3> dst, zisa::array_const_view<T, 3> src, T pad_value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;
  const auto src_shape = src.shape();
  const auto dst_shape = dst.shape();
  const int idx_src = zisa::row_major<3>::linear_index(src_shape, i, j, k);
  const int idx_dst = zisa::row_major<3>::linear_index(dst_shape, i, j, k);
  if (i < src_shape[0] && j < src_shape[1] && k < src_shape[2]) {
    dst[idx_dst] = src[idx_src];
  }
  else if (i < dst_shape[0] && j < dst_shape[1] && k < dst_shape[2]) {
    dst[idx_dst] = pad_value;
  }
}


template<typename T>
__global__ void copy_from_padded_cuda_kernel(zisa::array_view<T, 1> dst, zisa::array_const_view<T, 1> src) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dst.shape(0)) {
    dst[idx] = src[idx];
  }
}


template<typename T>
__global__ void copy_from_padded_cuda_kernel(zisa::array_view<T, 2> dst, zisa::array_const_view<T, 2> src) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const auto src_shape = src.shape();
  const auto dst_shape = dst.shape();
  const int idx_src = zisa::row_major<2>::linear_index(src_shape, i, j);
  const int idx_dst = zisa::row_major<2>::linear_index(dst_shape, i, j);
  if (i < dst_shape[0] && j < dst_shape[1]) {
    dst[idx_dst] = src[idx_src];
  }
}


template<typename T>
__global__ void copy_from_padded_cuda_kernel(zisa::array_view<T, 3> dst, zisa::array_const_view<T, 3> src) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;
  const auto src_shape = src.shape();
  const auto dst_shape = dst.shape();
  const int idx_src = zisa::row_major<3>::linear_index(src_shape, i, j, k);
  const int idx_dst = zisa::row_major<3>::linear_index(dst_shape, i, j, k);
  if (i < dst_shape[0] && j < dst_shape[1] && k < dst_shape[2]) {
    dst[idx_dst] = src[idx_src];
  }
}


template<typename T>
void copy_to_padded_cuda(const zisa::array_view<T, 1> &dst, const zisa::array_const_view<T, 1> &src, const T& pad_value) {
  assert(src.memory_location() == zisa::device_type::cuda);
  assert(dst.memory_location() == zisa::device_type::cuda);
  assert(dst.shape(0) >= src.shape(0));
  const int thread_dims = 1024;
  const int block_dims = zisa::min(zisa::div_up(static_cast<int>(dst.shape(0)), thread_dims), 1024);
  copy_to_padded_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, pad_value);
}


template<typename T>
void copy_to_padded_cuda(const zisa::array_view<T, 2> &dst, const zisa::array_const_view<T, 2> &src, const T& pad_value) {
  assert(src.memory_location() == zisa::device_type::cuda);
  assert(dst.memory_location() == zisa::device_type::cuda);
  assert(dst.shape(0) >= src.shape(0));
  assert(dst.shape(1) >= src.shape(1));
  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(zisa::min(zisa::div_up(static_cast<int>(dst.shape(0)), thread_dims.x), 32),
			zisa::min(zisa::div_up(static_cast<int>(dst.shape(1)), thread_dims.y), 32),
			1);
  copy_to_padded_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, pad_value);
}


template<typename T>
void copy_to_padded_cuda(const zisa::array_view<T, 3> &dst, const zisa::array_const_view<T, 3> &src, const T& pad_value) {
  assert(src.memory_location() == zisa::device_type::cuda);
  assert(dst.memory_location() == zisa::device_type::cuda);
  assert(dst.shape(0) >= src.shape(0));
  assert(dst.shape(1) >= src.shape(1));
  assert(dst.shape(2) >= src.shape(2));
  const dim3 thread_dims(8, 8, 8);
  const dim3 block_dims(zisa::min(zisa::div_up(static_cast<int>(dst.shape(0)), thread_dims.x), 8),
			zisa::min(zisa::div_up(static_cast<int>(dst.shape(1)), thread_dims.y), 8),
			zisa::min(zisa::div_up(static_cast<int>(dst.shape(2)), thread_dims.z), 8));
  copy_to_padded_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, pad_value);
}


template<typename T>
void copy_from_padded_cuda(const zisa::array_view<T, 1> &dst, const zisa::array_const_view<T, 1> &src) {
  assert(src.memory_location() == zisa::device_type::cuda);
  assert(dst.memory_location() == zisa::device_type::cuda);
  assert(dst.shape(0) <= src.shape(0));
  const int thread_dims = 1024;
  const int block_dims = zisa::min(zisa::div_up(static_cast<int>(dst.shape(0)), thread_dims), 1024);
  copy_from_padded_cuda_kernel<<<block_dims, thread_dims>>>(dst, src);
}


template<typename T>
void copy_from_padded_cuda(const zisa::array_view<T, 2> &dst, const zisa::array_const_view<T, 2> &src) {
  assert(src.memory_location() == zisa::device_type::cuda);
  assert(dst.memory_location() == zisa::device_type::cuda);
  assert(dst.shape(0) <= src.shape(0));
  assert(dst.shape(1) <= src.shape(1));
  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(zisa::min(zisa::div_up(static_cast<int>(dst.shape(0)), thread_dims.x), 32),
			zisa::min(zisa::div_up(static_cast<int>(dst.shape(1)), thread_dims.y), 32),
			1);
  copy_from_padded_cuda_kernel<<<block_dims, thread_dims>>>(dst, src);
}


template<typename T>
void copy_from_padded_cuda(const zisa::array_view<T, 3> &dst, const zisa::array_const_view<T, 3> &src) {
  assert(src.memory_location() == zisa::device_type::cuda);
  assert(dst.memory_location() == zisa::device_type::cuda);
  assert(dst.shape(0) <= src.shape(0));
  assert(dst.shape(1) <= src.shape(1));
  assert(dst.shape(2) <= src.shape(2));
  const dim3 thread_dims(8, 8, 8);
  const dim3 block_dims(zisa::min(zisa::div_up(static_cast<int>(dst.shape(0)), thread_dims.x), 8),
			zisa::min(zisa::div_up(static_cast<int>(dst.shape(1)), thread_dims.y), 8),
			zisa::min(zisa::div_up(static_cast<int>(dst.shape(2)), thread_dims.z), 8));
  copy_from_padded_cuda_kernel<<<block_dims, thread_dims>>>(dst, src);
}


}



#endif
