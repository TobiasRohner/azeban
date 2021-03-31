#ifndef SCALE_CUDA_IMPL_H_
#define SCALE_CUDA_IMPL_H_

#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename Scalar>
__global__ void scale_cuda_kernel(Scalar a, zisa::array_view<Scalar, 1> x) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < x.shape(0)) {
    x[i] *= a;
  }
}

template <typename Scalar>
void scale_cuda(const Scalar &a, const zisa::array_view<Scalar, 1> &x) {
  const int thread_dims = 1024;
  const int block_dims
      = zisa::div_up(static_cast<int>(x.shape(0)), thread_dims);
  scale_cuda_kernel<<<block_dims, thread_dims>>>(a, x);
  ZISA_CHECK_CUDA_DEBUG;
}

}

#endif
