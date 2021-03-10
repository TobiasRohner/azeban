#include <azeban/config.hpp>
#include <azeban/copy_padded.hpp>
#include <zisa/config.hpp>

namespace azeban {

__global__ void
copy_to_padded_cuda_kernel(zisa::array_view<complex_t, 1> dst,
                           zisa::array_const_view<complex_t, 1> src,
                           complex_t pad_value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < src.shape(0)) {
    dst[idx] = src[idx];
  } else if (idx < dst.shape(0)) {
    dst[idx] = pad_value;
  }
}

__global__ void
copy_to_padded_cuda_kernel(zisa::array_view<complex_t, 2> dst,
                           zisa::array_const_view<complex_t, 2> src,
                           complex_t pad_value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const auto src_shape = src.shape();
  const auto dst_shape = dst.shape();
  const int idx_dst = zisa::row_major<2>::linear_index(dst_shape, i, j);
  if (j < src_shape[1]) {
    if (i < dst_shape[0]) {
      if (i < src_shape[0] / 2 + 1) {
        const int idx_src = zisa::row_major<2>::linear_index(src_shape, i, j);
        dst[idx_dst] = src[idx_src];
      } else if (i < src_shape[0] / 2 + 1 + dst_shape[0] - src_shape[0]) {
        dst[idx_dst] = pad_value;
      } else {
        const int idx_src = zisa::row_major<2>::linear_index(
            src_shape, i + src_shape[0] - dst_shape[0], j);
        dst[idx_dst] = src[idx_src];
      }
    }
  } else if (i < dst_shape[0] && j < dst_shape[1]) {
    dst[idx_dst] = pad_value;
  }
}

__global__ void
copy_to_padded_cuda_kernel(zisa::array_view<complex_t, 3> dst,
                           zisa::array_const_view<complex_t, 3> src,
                           complex_t pad_value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;
  const auto src_shape = src.shape();
  const auto dst_shape = dst.shape();
  const int idx_dst = zisa::row_major<3>::linear_index(dst_shape, i, j, k);
  int i_src, j_src;

  if (i >= dst_shape[0] || j >= dst_shape[1] || k >= dst_shape[2]) {
    return;
  }

  if (k >= dst_shape[2]) {
    dst[idx_dst] = pad_value;
    return;
  }

  if (j < src_shape[1] / 2 + 1) {
    j_src = j;
  } else if (j < src_shape[1] / 2 + 1 + dst_shape[1] - src_shape[1]) {
    dst[idx_dst] = pad_value;
    return;
  } else {
    j_src = j + src_shape[1] - dst_shape[1];
  }

  if (i < src_shape[0] / 2 + 1) {
    i_src = i;
  } else if (i < src_shape[0] / 2 + 1 + dst_shape[0] - src_shape[0]) {
    dst[idx_dst] = pad_value;
    return;
  } else {
    i_src = i + src_shape[0] - dst_shape[0];
  }

  const int idx_src
      = zisa::row_major<3>::linear_index(src_shape, i_src, j_src, k);
  dst[idx_dst] = src[idx_src];
}

void copy_to_padded_cuda(const zisa::array_view<complex_t, 1> &dst,
                         const zisa::array_const_view<complex_t, 1> &src,
                         const complex_t &pad_value) {
  assert(src.memory_location() == zisa::device_type::cuda);
  assert(dst.memory_location() == zisa::device_type::cuda);
  assert(dst.shape(0) >= src.shape(0));
  const int thread_dims = 1024;
  const int block_dims = zisa::min(
      zisa::div_up(static_cast<int>(dst.shape(0)), thread_dims), 1024);
  copy_to_padded_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, pad_value);
}

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
  copy_to_padded_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, pad_value);
}

void copy_to_padded_cuda(const zisa::array_view<complex_t, 3> &dst,
                         const zisa::array_const_view<complex_t, 3> &src,
                         const complex_t &pad_value) {
  LOG_ERR("Not yet implemented");
  assert(src.memory_location() == zisa::device_type::cuda);
  assert(dst.memory_location() == zisa::device_type::cuda);
  assert(dst.shape(0) >= src.shape(0));
  assert(dst.shape(1) >= src.shape(1));
  assert(dst.shape(2) >= src.shape(2));
  const dim3 thread_dims(8, 8, 8);
  const dim3 block_dims(
      zisa::min(zisa::div_up(static_cast<int>(dst.shape(0)), thread_dims.x),
                1024),
      zisa::min(zisa::div_up(static_cast<int>(dst.shape(1)), thread_dims.y),
                1024),
      zisa::min(zisa::div_up(static_cast<int>(dst.shape(2)), thread_dims.z),
                1024));
  copy_to_padded_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, pad_value);
}

}
