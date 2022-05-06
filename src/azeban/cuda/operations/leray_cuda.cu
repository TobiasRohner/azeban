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
#include <azeban/config.hpp>
#include <azeban/cuda/operations/leray_cuda.hpp>

namespace azeban {

__global__ void leray_cuda_kernel(zisa::array_view<complex_t, 3> u_hat) {
  const unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long N_phys = u_hat.shape(1);
  const unsigned long N_fourier = N_phys / 2 + 1;

  if (i < u_hat.shape(1) && j < u_hat.shape(2)) {
    long i_ = i;
    if (i_ >= N_fourier) {
      i_ -= N_phys;
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j;
    const real_t absk2 = k1 * k1 + k2 * k2;
    const complex_t u1_hat = u_hat(0, i, j);
    const complex_t u2_hat = u_hat(1, i, j);
    u_hat(0, i, j) = absk2 == 0 ? 0.
                                : (1. - (k1 * k1) / absk2) * u1_hat
                                      + (0. - (k1 * k2) / absk2) * u2_hat;
    u_hat(1, i, j) = absk2 == 0 ? 0.
                                : (0. - (k2 * k1) / absk2) * u1_hat
                                      + (1. - (k2 * k2) / absk2) * u2_hat;
  }
}

__global__ void leray_cuda_kernel(zisa::array_view<complex_t, 4> u_hat) {
  const unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long k = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned long N_phys = u_hat.shape(1);
  const unsigned long N_fourier = N_phys / 2 + 1;

  if (i < u_hat.shape(1) && j < u_hat.shape(2) && k < u_hat.shape(3)) {
    long i_ = i;
    if (i_ >= N_fourier) {
      i_ -= N_phys;
    }
    long j_ = j;
    if (j_ >= N_fourier) {
      j_ -= N_phys;
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t k3 = 2 * zisa::pi * k;
    const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;
    const complex_t u1_hat = u_hat(0, i, j, k);
    const complex_t u2_hat = u_hat(1, i, j, k);
    const complex_t u3_hat = u_hat(2, i, j, k);
    u_hat(0, i, j, k) = absk2 == 0 ? 0.
                                   : (1. - (k1 * k1) / absk2) * u1_hat
                                         + (0. - (k1 * k2) / absk2) * u2_hat
                                         + (0. - (k1 * k3) / absk2) * u3_hat;
    u_hat(1, i, j, k) = absk2 == 0 ? 0.
                                   : (0. - (k2 * k1) / absk2) * u1_hat
                                         + (1. - (k2 * k2) / absk2) * u2_hat
                                         + (0. - (k2 * k3) / absk2) * u3_hat;
    u_hat(2, i, j, k) = absk2 == 0 ? 0.
                                   : (0. - (k3 * k1) / absk2) * u1_hat
                                         + (0. - (k3 * k2) / absk2) * u2_hat
                                         + (1. - (k3 * k3) / absk2) * u3_hat;
  }
}

void leray_cuda(const zisa::array_view<complex_t, 3> &u_hat) {
  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(u_hat.shape(1),
                   zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(u_hat.shape(2),
                   zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      1);
  leray_cuda_kernel<<<block_dims, thread_dims>>>(u_hat);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

void leray_cuda(const zisa::array_view<complex_t, 4> &u_hat) {
  const dim3 thread_dims(4, 4, 32);
  const dim3 block_dims(
      zisa::div_up(u_hat.shape(1),
                   zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(u_hat.shape(2),
                   zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      zisa::div_up(u_hat.shape(3),
                   zisa::integer_cast<zisa::int_t>(thread_dims.z)));
  leray_cuda_kernel<<<block_dims, thread_dims>>>(u_hat);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

__global__ void leray_cuda_mpi_kernel(zisa::array_view<complex_t, 3> u_hat,
                                      long k_start) {
  const unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long N_phys = u_hat.shape(2);
  const unsigned long N_fourier = N_phys / 2 + 1;

  if (i < u_hat.shape(1) && j < u_hat.shape(2)) {
    long i_ = i + k_start;
    if (i_ >= N_fourier) {
      i_ -= N_phys;
    }
    long j_ = j;
    if (j_ >= N_fourier) {
      j_ -= N_phys;
    }
    const real_t k2 = 2 * zisa::pi * i_;
    const real_t k1 = 2 * zisa::pi * j_;
    const real_t absk2 = k1 * k1 + k2 * k2;
    const complex_t u1_hat = u_hat(0, i, j);
    const complex_t u2_hat = u_hat(1, i, j);
    u_hat(0, i, j) = absk2 == 0 ? 0.
                                : (1. - (k1 * k1) / absk2) * u1_hat
                                      + (0. - (k1 * k2) / absk2) * u2_hat;
    u_hat(1, i, j) = absk2 == 0 ? 0.
                                : (0. - (k2 * k1) / absk2) * u1_hat
                                      + (1. - (k2 * k2) / absk2) * u2_hat;
  }
}

__global__ void leray_cuda_mpi_kernel(zisa::array_view<complex_t, 4> u_hat,
                                      long k_start) {
  const unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long k = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned long N_phys = u_hat.shape(3);
  const unsigned long N_fourier = N_phys / 2 + 1;

  if (i < u_hat.shape(1) && j < u_hat.shape(2) && k < u_hat.shape(3)) {
    long i_ = i + k_start;
    if (i_ >= N_fourier) {
      i_ -= N_phys;
    }
    long j_ = j;
    if (j_ >= N_fourier) {
      j_ -= N_phys;
    }
    long k_ = k;
    if (k_ >= N_fourier) {
      k_ -= N_phys;
    }
    const real_t k3 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t k1 = 2 * zisa::pi * k_;
    const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;
    const complex_t u1_hat = u_hat(0, i, j, k);
    const complex_t u2_hat = u_hat(1, i, j, k);
    const complex_t u3_hat = u_hat(2, i, j, k);
    u_hat(0, i, j, k) = absk2 == 0 ? 0.
                                   : (1. - (k1 * k1) / absk2) * u1_hat
                                         + (0. - (k1 * k2) / absk2) * u2_hat
                                         + (0. - (k1 * k3) / absk2) * u3_hat;
    u_hat(1, i, j, k) = absk2 == 0 ? 0.
                                   : (0. - (k2 * k1) / absk2) * u1_hat
                                         + (1. - (k2 * k2) / absk2) * u2_hat
                                         + (0. - (k2 * k3) / absk2) * u3_hat;
    u_hat(2, i, j, k) = absk2 == 0 ? 0.
                                   : (0. - (k3 * k1) / absk2) * u1_hat
                                         + (0. - (k3 * k2) / absk2) * u2_hat
                                         + (1. - (k3 * k3) / absk2) * u3_hat;
  }
}

void leray_cuda_mpi(const zisa::array_view<complex_t, 3> &u_hat, long k_start) {
  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(u_hat.shape(1),
                   zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(u_hat.shape(2),
                   zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      1);
  leray_cuda_mpi_kernel<<<block_dims, thread_dims>>>(u_hat, k_start);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

void leray_cuda_mpi(const zisa::array_view<complex_t, 4> &u_hat, long k_start) {
  const dim3 thread_dims(4, 4, 32);
  const dim3 block_dims(
      zisa::div_up(u_hat.shape(1),
                   zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(u_hat.shape(2),
                   zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      zisa::div_up(u_hat.shape(3),
                   zisa::integer_cast<zisa::int_t>(thread_dims.z)));
  leray_cuda_mpi_kernel<<<block_dims, thread_dims>>>(u_hat, k_start);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

}
