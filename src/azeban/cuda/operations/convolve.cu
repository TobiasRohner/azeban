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
#include <azeban/cuda/operations/convolve_cuda.hpp>

namespace azeban {

__global__ void scale_and_square_kernel(zisa::array_view<real_t, 1> u,
                                        real_t scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < u.shape(0)) {
    const real_t ui_scaled = scale * u[i];
    u[i] = ui_scaled * ui_scaled;
  }
}

void scale_and_square_cuda(const zisa::array_view<real_t, 1> &u, real_t scale) {
  const int thread_dims = 1024;
  const int block_dims = zisa::min(
      zisa::div_up(static_cast<int>(u.shape(0)), thread_dims), 1024);
  scale_and_square_kernel<<<block_dims, thread_dims>>>(u, scale);
  ZISA_CHECK_CUDA_DEBUG;
  cudaDeviceSynchronize();
}

}
