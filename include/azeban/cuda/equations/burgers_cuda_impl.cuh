#ifndef BURGERS_CUDA_IMPL_H_
#define BURGERS_CUDA_IMPL_H_

#include "burgers_cuda.hpp"
#include <azeban/config.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/comparison.hpp>
#include <zisa/config.hpp>



namespace azeban {


template<typename SpectralViscosity>
__global__ void burgers_cuda_kernel(zisa::array_view<complex_t, 1> u,
				    zisa::array_const_view<complex_t, 1> u_squared,
				    SpectralViscosity visc) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < u.shape(0)) {
    const real_t k_ = static_cast<real_t>(k) * zisa::pi / u.shape(0);
    const real_t v = visc.eval(k_);
    u[k].x = k_*u_squared[k].y/2 + v*u[k].x;
    u[k].y = -k_*u_squared[k].x/2 + v*u[k].y;
  }
}


template<typename SpectralViscosity>
void burgers_cuda(const zisa::array_view<complex_t, 1> &u,
		  const zisa::array_const_view<complex_t, 1> &u_squared,
		  const SpectralViscosity &visc) {
  const int thread_dims = 1024;
  const int block_dims = zisa::min(zisa::div_up(static_cast<int>(u.shape(0)), thread_dims), 1024);
  burgers_cuda_kernel<<<thread_dims, block_dims>>>(u, u_squared, visc);
}


}



#endif
