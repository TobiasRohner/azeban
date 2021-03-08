#ifndef AXPBY_CUDA_IMPL_H_
#define AXPBY_CUDA_IMPL_H_

#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename Scalar>
__global__ void axpby_cuda_kernel(Scalar a,
                                  zisa::array_const_view<Scalar, 1> x,
                                  Scalar b,
                                  zisa::array_view<Scalar, 1> y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < x.shape(0)) {
    y[i] = a * x[i] + b * y[i];
  }
}

template <typename Scalar>
void axpby_cuda(const Scalar &a,
                const zisa::array_const_view<Scalar, 1> &x,
                const Scalar &b,
                const zisa::array_view<Scalar, 1> &y) {
  assert(x.shape() == y.shape());
  const int thread_dims = 1024;
  const int block_dims = zisa::min(
      zisa::div_up(static_cast<int>(x.shape(0)), thread_dims), 1024);
  axpby_cuda_kernel<<<block_dims, thread_dims>>>(a, x, b, y);
}

}

#endif