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
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

}

#endif
