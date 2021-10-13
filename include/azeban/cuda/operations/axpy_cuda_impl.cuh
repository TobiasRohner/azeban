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
#ifndef AXPY_CUDA_IMPL_H_
#define AXPY_CUDA_IMPL_H_

#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename Scalar>
__global__ void axpy_cuda_kernel(Scalar a,
                                 zisa::array_const_view<Scalar, 1> x,
                                 zisa::array_view<Scalar, 1> y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < x.shape(0)) {
    y[i] += a * x[i];
  }
}

template <typename Scalar>
void axpy_cuda(const Scalar &a,
               const zisa::array_const_view<Scalar, 1> &x,
               const zisa::array_view<Scalar, 1> &y) {
  assert(x.shape() == y.shape());
  const int thread_dims = 1024;
  const int block_dims
      = zisa::div_up(static_cast<int>(x.shape(0)), thread_dims);
  axpy_cuda_kernel<<<block_dims, thread_dims>>>(a, x, y);
  ZISA_CHECK_CUDA_DEBUG;
  cudaDeviceSynchronize();
}

}

#endif
