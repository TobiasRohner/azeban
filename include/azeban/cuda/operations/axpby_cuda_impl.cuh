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
  const zisa::int_t i = blockIdx.x * blockDim.x + threadIdx.x;
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
  const int block_dims
      = zisa::div_up(x.shape(0), zisa::integer_cast<zisa::int_t>(thread_dims));
  axpby_cuda_kernel<<<block_dims, thread_dims>>>(a, x, b, y);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

}

#endif
