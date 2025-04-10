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
#ifndef CLAMP_CUDA_IMPL_H_
#define CLAMP_CUDA_IMPL_H_

#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename Scalar>
__global__ void clamp_cuda_kernel(zisa::array_view<Scalar, 1> x, real_t a) {
  using azeban::abs;
  using zisa::abs;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < x.shape(0)) {
    const Scalar &value = x[i];
    x[i] = value / zisa::max(real_t{1}, static_cast<real_t>(abs(value) / a));
  }
}

template <typename Scalar>
void clamp_cuda(const zisa::array_view<Scalar, 1> &x, real_t a) {
  const int thread_dims = 1024;
  const int block_dims
      = zisa::div_up(static_cast<int>(x.shape(0)), thread_dims);
  clamp_cuda_kernel<<<block_dims, thread_dims>>>(x, a);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

}

#endif
