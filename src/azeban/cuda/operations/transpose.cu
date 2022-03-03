#include <azeban/cuda/operations/transpose.hpp>

namespace azeban {

__global__ void
transpose_cuda_preprocess_kernel(zisa::array_const_view<complex_t, 3> from,
                                 zisa::array_view<complex_t, 3> sendbuf,
                                 zisa::shape_t<3> shape,
                                 zisa::int_t j_offset) {
  __shared__ complex_t tile[32][33]; // Pad to 33 to prevent bank conflicts
  for (int d = 0; d < shape[0]; ++d) {
    zisa::int_t x = 32 * blockIdx.x + threadIdx.x + j_offset;
    zisa::int_t y = 32 * blockIdx.y + threadIdx.y;
    if (x < shape[2]) {
      const int j_max = zisa::min(32, zisa::integer_cast<int>(shape[1] - y));
      for (int j = 0; j < j_max; j += 8) {
        tile[threadIdx.y + j][threadIdx.x] = from(d, y + j, x);
      }
    }
    __syncthreads();
    x = 32 * blockIdx.y + threadIdx.x;
    y = 32 * blockIdx.x + threadIdx.y;
    if (x < shape[1]) {
      const int j_max = zisa::min(32, zisa::integer_cast<int>(shape[2] - y));
      for (int j = 0; j < j_max; j += 8) {
        sendbuf(d, y + j, x) = tile[threadIdx.x][threadIdx.y + j];
      }
    }
    __syncthreads();
  }
}

__global__ void
transpose_cuda_preprocess_kernel(zisa::array_const_view<complex_t, 4> from,
                                 zisa::array_view<complex_t, 4> sendbuf,
                                 zisa::shape_t<4> shape,
                                 zisa::int_t k_offset) {
  __shared__ complex_t tile[32][33]; // Pad to 33 to prevent bank conflicts
  for (int d = 0; d < shape[0]; ++d) {
    for (int k = 0; k < shape[2]; ++k) {
      zisa::int_t x = 32 * blockIdx.x + threadIdx.x;
      zisa::int_t y = 32 * blockIdx.y + threadIdx.y;
      if (x < shape[3] && y < shape[1]) {
        tile[threadIdx.y][threadIdx.x] = from(d, y, k, x + k_offset);
      }
      __syncthreads();
      x = 32 * blockIdx.y + threadIdx.x;
      y = 32 * blockIdx.x + threadIdx.y;
      if (x < shape[1] && y < shape[3]) {
        sendbuf(d, y, k, x) = tile[threadIdx.x][threadIdx.y];
      }
      __syncthreads();
    }
  }
}

__global__ void
transpose_cuda_postprocess_kernel(zisa::array_const_view<complex_t, 3> recvbuf,
                                  zisa::array_view<complex_t, 3> to,
                                  zisa::shape_t<3> shape,
                                  zisa::int_t j_offset) {
  const zisa::int_t i = blockDim.y * blockIdx.y + threadIdx.y;
  const zisa::int_t j = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= shape[1] || j >= shape[2]) {
    return;
  }
  for (int d = 0; d < shape[0]; ++d) {
    to(d, i, j + j_offset) = recvbuf(d, i, j);
  }
}

__global__ void
transpose_cuda_postprocess_kernel(zisa::array_const_view<complex_t, 4> recvbuf,
                                  zisa::array_view<complex_t, 4> to,
                                  zisa::shape_t<4> shape,
                                  zisa::int_t k_offset) {
  const zisa::int_t i = blockDim.y * blockIdx.y + threadIdx.y;
  const zisa::int_t j = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= shape[1] || j >= shape[3]) {
    return;
  }
  for (int d = 0; d < shape[0]; ++d) {
    for (int k = 0; k < shape[2]; ++k) {
      to(d, i, k, j + k_offset) = recvbuf(d, i, k, j);
    }
  }
}

void transpose_cuda_preprocess(const zisa::array_const_view<complex_t, 3> &from,
                               const zisa::array_view<complex_t, 4> &sendbuf,
                               const zisa::shape_t<3> *from_shapes,
                               const zisa::shape_t<3> *to_shapes,
                               int rank) {
  const int size = sendbuf.shape(0);
  zisa::int_t j_offset = 0;
  for (int r = 0; r < size; ++r) {
    const zisa::shape_t<3> sendbuf_view_shape{
        sendbuf.shape(1), sendbuf.shape(2), sendbuf.shape(3)};
    zisa::array_view<complex_t, 3> sendbuf_view(
        sendbuf_view_shape,
        sendbuf.raw() + r * zisa::product(sendbuf_view_shape),
        sendbuf.memory_location());
    const zisa::shape_t<3> block_shape{
        from_shapes[rank][0], from_shapes[rank][1], to_shapes[r][1]};
    const dim3 thread_dims(32, 8, 1);
    const dim3 block_dims(
        zisa::min(zisa::div_up(block_shape[2],
                               zisa::integer_cast<zisa::int_t>(thread_dims.x)),
                  static_cast<zisa::int_t>(1024)),
        zisa::min(
            zisa::div_up(block_shape[1],
                         zisa::integer_cast<zisa::int_t>(4 * thread_dims.y)),
            static_cast<zisa::int_t>(1024)),
        1);
    transpose_cuda_preprocess_kernel<<<block_dims, thread_dims>>>(
        from, sendbuf_view, block_shape, j_offset);
    cudaDeviceSynchronize();
    ZISA_CHECK_CUDA_DEBUG;
    j_offset += to_shapes[r][1];
  }
}

void transpose_cuda_preprocess(const zisa::array_const_view<complex_t, 4> &from,
                               const zisa::array_view<complex_t, 5> &sendbuf,
                               const zisa::shape_t<4> *from_shapes,
                               const zisa::shape_t<4> *to_shapes,
                               int rank) {
  const int size = sendbuf.shape(0);
  zisa::int_t k_offset = 0;
  for (int r = 0; r < size; ++r) {
    const zisa::shape_t<4> sendbuf_view_shape{
        sendbuf.shape(1), sendbuf.shape(2), sendbuf.shape(3), sendbuf.shape(4)};
    zisa::array_view<complex_t, 4> sendbuf_view(
        sendbuf_view_shape,
        sendbuf.raw() + r * zisa::product(sendbuf_view_shape),
        sendbuf.memory_location());
    const zisa::shape_t<4> block_shape{from_shapes[rank][0],
                                       from_shapes[rank][1],
                                       from_shapes[rank][2],
                                       to_shapes[r][1]};
    const dim3 thread_dims(32, 32, 1);
    const dim3 block_dims(
        zisa::min(zisa::div_up(block_shape[3],
                               zisa::integer_cast<zisa::int_t>(thread_dims.x)),
                  static_cast<zisa::int_t>(1024)),
        zisa::min(zisa::div_up(block_shape[1],
                               zisa::integer_cast<zisa::int_t>(thread_dims.y)),
                  static_cast<zisa::int_t>(1024)),
        1);
    transpose_cuda_preprocess_kernel<<<block_dims, thread_dims>>>(
        from, sendbuf_view, block_shape, k_offset);
    cudaDeviceSynchronize();
    ZISA_CHECK_CUDA_DEBUG;
    k_offset += to_shapes[r][1];
  }
}

void transpose_cuda_postprocess(
    const zisa::array_const_view<complex_t, 4> &recvbuf,
    const zisa::array_view<complex_t, 3> &to,
    const zisa::shape_t<3> *from_shapes,
    const zisa::shape_t<3> *to_shapes,
    int rank) {
  const int size = recvbuf.shape(0);
  zisa::int_t j_offset = 0;
  for (int r = 0; r < size; ++r) {
    const zisa::shape_t<3> recvbuf_view_shape{
        recvbuf.shape(1), recvbuf.shape(2), recvbuf.shape(3)};
    zisa::array_const_view<complex_t, 3> recvbuf_view(
        recvbuf_view_shape,
        recvbuf.raw() + r * zisa::product(recvbuf_view_shape),
        recvbuf.memory_location());
    const zisa::shape_t<3> block_shape{
        to_shapes[rank][0], to_shapes[rank][1], from_shapes[r][1]};
    const dim3 thread_dims(32, 32, 1);
    const dim3 block_dims(
        zisa::min(zisa::div_up(block_shape[3],
                               zisa::integer_cast<zisa::int_t>(thread_dims.x)),
                  static_cast<zisa::int_t>(1024)),
        zisa::min(zisa::div_up(block_shape[1],
                               zisa::integer_cast<zisa::int_t>(thread_dims.y)),
                  static_cast<zisa::int_t>(1024)),
        1);
    transpose_cuda_postprocess_kernel<<<block_dims, thread_dims>>>(
        recvbuf_view, to, block_shape, j_offset);
    cudaDeviceSynchronize();
    ZISA_CHECK_CUDA_DEBUG;
    j_offset += from_shapes[r][1];
  }
}

void transpose_cuda_postprocess(
    const zisa::array_const_view<complex_t, 5> &recvbuf,
    const zisa::array_view<complex_t, 4> &to,
    const zisa::shape_t<4> *from_shapes,
    const zisa::shape_t<4> *to_shapes,
    int rank) {
  const int size = recvbuf.shape(0);
  zisa::int_t k_offset = 0;
  for (int r = 0; r < size; ++r) {
    const zisa::shape_t<4> recvbuf_view_shape{
        recvbuf.shape(1), recvbuf.shape(2), recvbuf.shape(3), recvbuf.shape(4)};
    zisa::array_const_view<complex_t, 4> recvbuf_view(
        recvbuf_view_shape,
        recvbuf.raw() + r * zisa::product(recvbuf_view_shape),
        recvbuf.memory_location());
    const zisa::shape_t<4> block_shape{to_shapes[rank][0],
                                       to_shapes[rank][1],
                                       to_shapes[rank][2],
                                       from_shapes[r][1]};
    const dim3 thread_dims(32, 32, 1);
    const dim3 block_dims(
        zisa::min(zisa::div_up(block_shape[3],
                               zisa::integer_cast<zisa::int_t>(thread_dims.x)),
                  static_cast<zisa::int_t>(1024)),
        zisa::min(zisa::div_up(block_shape[1],
                               zisa::integer_cast<zisa::int_t>(thread_dims.y)),
                  static_cast<zisa::int_t>(1024)),
        1);
    transpose_cuda_postprocess_kernel<<<block_dims, thread_dims>>>(
        recvbuf_view, to, block_shape, k_offset);
    cudaDeviceSynchronize();
    ZISA_CHECK_CUDA_DEBUG;
    k_offset += from_shapes[r][1];
  }
}

}
