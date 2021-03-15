#ifndef BURGERS_CUDA_IMPL_H_
#define BURGERS_CUDA_IMPL_H_

#include "burgers_cuda.hpp"
#include <azeban/config.hpp>
#include <zisa/config.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/comparison.hpp>

namespace azeban {

template <typename SpectralViscosity>
__global__ void
burgers_cuda_kernel(zisa::array_view<complex_t, 2> u,
                    zisa::array_const_view<complex_t, 2> u_squared,
                    SpectralViscosity visc) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < u.shape(1)) {
    const real_t k_ = 2 * zisa::pi * k;
    const real_t v = visc.eval(k_);
    u[k] = complex_t(0, -k_ / 2) * u_squared[k] + v * u[k];
  }
}

template <typename SpectralViscosity>
void burgers_cuda(const zisa::array_view<complex_t, 2> &u,
                  const zisa::array_const_view<complex_t, 2> &u_squared,
                  const SpectralViscosity &visc) {
  const int thread_dims = 1024;
  const int block_dims
      = zisa::div_up(static_cast<int>(u.shape(1)), thread_dims);
  burgers_cuda_kernel<<<block_dims, thread_dims>>>(u, u_squared, visc);
  ZISA_CHECK_CUDA_DEBUG;
}

}

#endif
