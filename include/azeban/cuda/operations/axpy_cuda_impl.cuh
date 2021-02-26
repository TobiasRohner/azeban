#ifndef AXPY_CUDA_IMPL_H_
#define AXPY_CUDA_IMPL_H_

#include <zisa/memory/array_view.hpp>
#include <zisa/math/basic_functions.hpp>



namespace azeban {


template<typename Scalar>
__global__ void axpy_cuda_kernel(Scalar a, zisa::array_const_view<Scalar, 1> x, zisa::array_view<Scalar, 1> y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < x.shape(0)) {
    y[i] = a * x[i] + y[i];
  }
}


template<typename Scalar>
void axpy_cuda(const Scalar &a, const zisa::array_const_view<Scalar, 1> &x, const zisa::array_view<Scalar, 1> &y) {
  assert(x.shape() == y.shape());
  const int thread_dims = 1024;
  const int block_dims = zisa::min(zisa::div_up(static_cast<int>(x.shape(0)), thread_dims), 1024);
  axpy_cuda_kernel<<<thread_dims, block_dims>>>(a, x, y);
}


}



#endif
