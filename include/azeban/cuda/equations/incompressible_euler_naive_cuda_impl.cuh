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
#ifndef INCOMPRESSIBLE_EULER_NAIVE_CUDA_IMPL_H_
#define INCOMPRESSIBLE_EULER_NAIVE_CUDA_IMPL_H_

#include "incompressible_euler_naive_cuda.hpp"

namespace azeban {

template <int Dim>
__global__ void incompressible_euler_naive_compute_B_cuda_kernel(
    zisa::array_view<real_t, Dim + 1> B,
    zisa::array_const_view<real_t, Dim + 1> u,
    Grid<Dim> grid) {}

template <>
__global__ void incompressible_euler_naive_compute_B_cuda_kernel<2>(
    zisa::array_view<real_t, 3> B,
    zisa::array_const_view<real_t, 3> u,
    Grid<2> grid) {
  const unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long stride = grid.N_phys_pad * grid.N_phys_pad;
  const unsigned long idx = i * grid.N_phys_pad + j;

  const real_t norm = 1.0 / (grid.N_phys * grid.N_phys_pad);
  if (i < grid.N_phys_pad && j < grid.N_phys_pad) {
    const real_t u1 = norm * u[0 * stride + idx];
    const real_t u2 = norm * u[1 * stride + idx];
    B[0 * stride + idx] = u1 * u1;
    B[1 * stride + idx] = u1 * u2;
    B[2 * stride + idx] = u2 * u2;
  }
}

template <>
__global__ void incompressible_euler_naive_compute_B_cuda_kernel<3>(
    zisa::array_view<real_t, 4> B,
    zisa::array_const_view<real_t, 4> u,
    Grid<3> grid) {
  const unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long k = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned long stride
      = grid.N_phys_pad * grid.N_phys_pad * grid.N_phys_pad;
  const unsigned long idx
      = i * grid.N_phys_pad * grid.N_phys_pad + j * grid.N_phys_pad + k;

  const real_t norm
      = 1.0 / (zisa::pow(grid.N_phys, 1.5) * zisa::pow(grid.N_phys_pad, 1.5));
  if (i < grid.N_phys_pad && j < grid.N_phys_pad && k < grid.N_phys_pad) {
    const real_t u1 = norm * u[0 * stride + idx];
    const real_t u2 = norm * u[1 * stride + idx];
    const real_t u3 = norm * u[2 * stride + idx];
    B[0 * stride + idx] = u1 * u1;
    B[1 * stride + idx] = u2 * u1;
    B[2 * stride + idx] = u2 * u2;
    B[3 * stride + idx] = u3 * u1;
    B[4 * stride + idx] = u3 * u2;
    B[5 * stride + idx] = u3 * u3;
  }
}

template <typename SpectralViscosity>
__global__ void incompressible_euler_naive_2d_cuda_kernel(
    zisa::array_const_view<complex_t, 3> B_hat,
    zisa::array_const_view<complex_t, 3> u_hat,
    zisa::array_view<complex_t, 3> dudt_hat,
    SpectralViscosity visc) {
  const unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long stride_B = B_hat.shape(1) * B_hat.shape(2);
  const unsigned long i_B
      = i >= u_hat.shape(1) / 2 + 1 ? B_hat.shape(1) - u_hat.shape(1) + i : i;
  const unsigned long idx_B = i_B * B_hat.shape(2) + j;
  const unsigned long stride_u = u_hat.shape(1) * u_hat.shape(2);
  const unsigned long idx_u = i * u_hat.shape(2) + j;

  if (i < u_hat.shape(1) && j < u_hat.shape(2)) {
    long i_ = i;
    if (i_ >= u_hat.shape(1) / 2 + 1) {
      i_ -= u_hat.shape(1);
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j;

    const complex_t B11_hat = B_hat[0 * stride_B + idx_B];
    const complex_t B12_hat = B_hat[1 * stride_B + idx_B];
    const complex_t B22_hat = B_hat[2 * stride_B + idx_B];
    const complex_t b1_hat
        = complex_t(0, k1) * B11_hat + complex_t(0, k2) * B12_hat;
    const complex_t b2_hat
        = complex_t(0, k1) * B12_hat + complex_t(0, k2) * B22_hat;

    const real_t absk2 = k1 * k1 + k2 * k2;
    const complex_t L1_hat
        = (1. - (k1 * k1) / absk2) * b1_hat + (0. - (k1 * k2) / absk2) * b2_hat;
    const complex_t L2_hat
        = (0. - (k2 * k1) / absk2) * b1_hat + (1. - (k2 * k2) / absk2) * b2_hat;

    const real_t v = visc.eval(zisa::sqrt(absk2));
    dudt_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + v * u_hat[0 * stride_u + idx_u];
    dudt_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + v * u_hat[1 * stride_u + idx_u];
  }
}

template <typename SpectralViscosity>
__global__ void incompressible_euler_naive_3d_cuda_kernel(
    zisa::array_const_view<complex_t, 4> B_hat,
    zisa::array_const_view<complex_t, 4> u_hat,
    zisa::array_view<complex_t, 4> dudt_hat,
    SpectralViscosity visc) {
  const unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long k = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned long stride_B
      = B_hat.shape(1) * B_hat.shape(2) * B_hat.shape(3);
  const unsigned long i_B
      = i >= u_hat.shape(1) / 2 + 1 ? B_hat.shape(1) - u_hat.shape(1) + i : i;
  const unsigned long j_B
      = j >= u_hat.shape(2) / 2 + 1 ? B_hat.shape(2) - u_hat.shape(2) + j : j;
  const unsigned long idx_B
      = i_B * B_hat.shape(2) * B_hat.shape(3) + j_B * B_hat.shape(3) + k;
  const unsigned long stride_u
      = u_hat.shape(1) * u_hat.shape(2) * u_hat.shape(3);
  const unsigned long idx_u
      = i * u_hat.shape(2) * u_hat.shape(3) + j * u_hat.shape(3) + k;

  if (i < u_hat.shape(1) && j < u_hat.shape(2) && k < u_hat.shape(3)) {
    long i_ = i;
    long j_ = j;
    if (i_ >= u_hat.shape(1) / 2 + 1) {
      i_ -= u_hat.shape(1);
    }
    if (j_ >= u_hat.shape(2) / 2 + 1) {
      i_ -= u_hat.shape(2);
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t k3 = 2 * zisa::pi * k;

    const complex_t B11_hat = B_hat[0 * stride_B + idx_B];
    const complex_t B21_hat = B_hat[1 * stride_B + idx_B];
    const complex_t B22_hat = B_hat[2 * stride_B + idx_B];
    const complex_t B31_hat = B_hat[3 * stride_B + idx_B];
    const complex_t B32_hat = B_hat[4 * stride_B + idx_B];
    const complex_t B33_hat = B_hat[5 * stride_B + idx_B];
    const complex_t b1_hat = complex_t(0, k1) * B11_hat
                             + complex_t(0, k2) * B21_hat
                             + complex_t(0, k3) * B31_hat;
    const complex_t b2_hat = complex_t(0, k1) * B21_hat
                             + complex_t(0, k2) * B22_hat
                             + complex_t(0, k3) * B32_hat;
    const complex_t b3_hat = complex_t(0, k1) * B31_hat
                             + complex_t(0, k2) * B32_hat
                             + complex_t(0, k3) * B33_hat;

    const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;
    const complex_t L1_hat = (1. - (k1 * k1) / absk2) * b1_hat
                             + (0. - (k1 * k2) / absk2) * b2_hat
                             + (0. - (k1 * k3) / absk2) * b3_hat;
    const complex_t L2_hat = (0. - (k2 * k1) / absk2) * b1_hat
                             + (1. - (k2 * k2) / absk2) * b2_hat
                             + (0. - (k2 * k3) / absk2) * b3_hat;
    const complex_t L3_hat = (0. - (k3 * k1) / absk2) * b1_hat
                             + (0. - (k3 * k2) / absk2) * b2_hat
                             + (1. - (k3 * k3) / absk2) * b3_hat;

    const real_t v = visc.eval(zisa::sqrt(absk2));
    dudt_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + v * u_hat[0 * stride_u + idx_u];
    dudt_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + v * u_hat[1 * stride_u + idx_u];
    dudt_hat[2 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L3_hat + v * u_hat[2 * stride_u + idx_u];
  }
}

template <int Dim>
void incompressible_euler_naive_compute_B_cuda(
    const zisa::array_view<real_t, Dim + 1> &B,
    const zisa::array_const_view<real_t, Dim + 1> &u,
    const Grid<Dim> &grid) {}

template <>
void incompressible_euler_naive_compute_B_cuda<2>(
    const zisa::array_view<real_t, 3> &B,
    const zisa::array_const_view<real_t, 3> &u,
    const Grid<2> &grid) {
  assert(B.memory_location() == zisa::device_type::cuda);
  assert(u.memory_location() == zisa::device_type::cuda);
  assert(B.shape(1) == u.shape(1));
  assert(B.shape(2) == u.shape(2));

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(u.shape(1), zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(u.shape(2), zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      1);
  incompressible_euler_naive_compute_B_cuda_kernel<2>
      <<<block_dims, thread_dims>>>(B, u, grid);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <>
void incompressible_euler_naive_compute_B_cuda<3>(
    const zisa::array_view<real_t, 4> &B,
    const zisa::array_const_view<real_t, 4> &u,
    const Grid<3> &grid) {
  assert(B.memory_location() == zisa::device_type::cuda);
  assert(u.memory_location() == zisa::device_type::cuda);
  assert(B.shape(1) == u.shape(1));
  assert(B.shape(2) == u.shape(2));
  assert(B.shape(3) == u.shape(3));
  const dim3 thread_dims(8, 8, 8);
  const dim3 block_dims(
      zisa::div_up(u.shape(1), zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(u.shape(2), zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      zisa::div_up(u.shape(3), zisa::integer_cast<zisa::int_t>(thread_dims.z)));
  incompressible_euler_naive_compute_B_cuda_kernel<3>
      <<<block_dims, thread_dims>>>(B, u, grid);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity>
void incompressible_euler_naive_2d_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &dudt_hat,
    const SpectralViscosity &visc) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);
  assert(dudt_hat.memory_location() == zisa::device_type::cuda);

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(u_hat.shape(1),
                   zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(u_hat.shape(2),
                   zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      1);

  incompressible_euler_naive_2d_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, dudt_hat, visc);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity>
void incompressible_euler_naive_3d_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);
  assert(dudt_hat.memory_location() == zisa::device_type::cuda);
  const dim3 thread_dims(8, 8, 8);
  const dim3 block_dims(
      zisa::div_up(u_hat.shape(1),
                   zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(u_hat.shape(2),
                   zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      zisa::div_up(u_hat.shape(3),
                   zisa::integer_cast<zisa::int_t>(thread_dims.z)));
  incompressible_euler_naive_3d_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, dudt_hat, visc);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

}

#endif
