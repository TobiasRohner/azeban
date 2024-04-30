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
#ifndef AZEBAN_CUDA_EQUATIONS_INCOMPRESSIBLE_EULER_CUDA_IMPL_HPP_
#define AZEBAN_CUDA_EQUATIONS_INCOMPRESSIBLE_EULER_CUDA_IMPL_HPP_

#include "incompressible_euler_cuda.hpp"
#include <azeban/equations/advection_functions.hpp>
#include <azeban/equations/incompressible_euler_functions.hpp>

namespace azeban {

template <int Dim>
__global__ void incompressible_euler_compute_B_cuda_kernel(
    zisa::array_view<real_t, Dim + 1> B,
    zisa::array_const_view<real_t, Dim + 1> u,
    Grid<Dim> grid) {}

template <>
__global__ void incompressible_euler_compute_B_cuda_kernel<2>(
    zisa::array_view<real_t, 3> B,
    zisa::array_const_view<real_t, 3> u,
    Grid<2> grid) {
  const unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = u.shape(1) * u.shape(2);
  const unsigned idx = i * u.shape(2) + j;

  const real_t norm = 1.0 / (grid.N_phys * grid.N_phys_pad);
  if (i < u.shape(1) && j < u.shape(2)) {
    const real_t u1 = norm * u[0 * stride + idx];
    const real_t u2 = norm * u[1 * stride + idx];
    incompressible_euler_2d_compute_B(stride, idx, u1, u2, B.raw());
  }
}

template <>
__global__ void incompressible_euler_compute_B_cuda_kernel<3>(
    zisa::array_view<real_t, 4> B,
    zisa::array_const_view<real_t, 4> u,
    Grid<3> grid) {
  const unsigned i = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = u.shape(1) * u.shape(2) * u.shape(3);
  const unsigned idx = i * u.shape(2) * u.shape(3) + j * u.shape(3) + k;

  const real_t norm
      = 1.0 / (zisa::pow(grid.N_phys, 1.5) * zisa::pow(grid.N_phys_pad, 1.5));
  if (i < u.shape(1) && j < u.shape(2) && k < u.shape(3)) {
    const real_t u1 = norm * u[0 * stride + idx];
    const real_t u2 = norm * u[1 * stride + idx];
    const real_t u3 = norm * u[2 * stride + idx];
    incompressible_euler_3d_compute_B(stride, idx, u1, u2, u3, B.raw());
  }
}

template <int Dim>
__global__ void incompressible_euler_compute_B_tracer_cuda_kernel(
    zisa::array_view<real_t, Dim + 1> B,
    zisa::array_const_view<real_t, Dim + 1> u,
    Grid<Dim> grid) {}

template <>
__global__ void incompressible_euler_compute_B_tracer_cuda_kernel<2>(
    zisa::array_view<real_t, 3> B,
    zisa::array_const_view<real_t, 3> u,
    Grid<2> grid) {
  const unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = u.shape(1) * u.shape(2);
  const unsigned idx = i * u.shape(2) + j;

  const real_t norm = 1.0 / (grid.N_phys * grid.N_phys_pad);
  if (i < u.shape(1) && j < u.shape(2)) {
    const real_t u1 = norm * u[0 * stride + idx];
    const real_t u2 = norm * u[1 * stride + idx];
    const real_t rho = norm * u[2 * stride + idx];
    incompressible_euler_2d_compute_B(stride, idx, u1, u2, B.raw());
    advection_2d_compute_B(stride, idx, rho, u1, u2, B.raw() + 3 * stride);
  }
}

template <>
__global__ void incompressible_euler_compute_B_tracer_cuda_kernel<3>(
    zisa::array_view<real_t, 4> B,
    zisa::array_const_view<real_t, 4> u,
    Grid<3> grid) {
  const unsigned i = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = u.shape(1) * u.shape(2) * u.shape(3);
  const unsigned idx = i * u.shape(2) * u.shape(3) + j * u.shape(3) + k;

  const real_t norm
      = 1.0 / (zisa::pow(grid.N_phys, 1.5) * zisa::pow(grid.N_phys_pad, 1.5));
  if (i < u.shape(1) && j < u.shape(2) && k < u.shape(3)) {
    const real_t u1 = norm * u[0 * stride + idx];
    const real_t u2 = norm * u[1 * stride + idx];
    const real_t u3 = norm * u[2 * stride + idx];
    const real_t rho = norm * u[3 * stride + idx];
    incompressible_euler_3d_compute_B(stride, idx, u1, u2, u3, B.raw());
    advection_3d_compute_B(stride, idx, rho, u1, u2, u3, B.raw() + 6 * stride);
  }
}

template <typename SpectralViscosity, typename Forcing>
__global__ void
incompressible_euler_2d_cuda_kernel(zisa::array_const_view<complex_t, 3> B_hat,
                                    zisa::array_const_view<complex_t, 3> u_hat,
                                    zisa::array_view<complex_t, 3> dudt_hat,
                                    SpectralViscosity visc,
                                    Forcing forcing,
                                    real_t t,
                                    real_t dt) {
  const unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride_B = B_hat.shape(1) * B_hat.shape(2);
  const unsigned i_B
      = i >= u_hat.shape(1) / 2 + 1 ? B_hat.shape(1) - u_hat.shape(1) + i : i;
  const unsigned idx_B = i_B * B_hat.shape(2) + j;
  const unsigned stride_u = u_hat.shape(1) * u_hat.shape(2);
  const unsigned idx_u = i * u_hat.shape(2) + j;

  if (i < u_hat.shape(1) && j < u_hat.shape(2)) {
    int i_ = i;
    if (i_ >= u_hat.shape(1) / 2 + 1) {
      i_ -= u_hat.shape(1);
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j;
    const real_t absk2 = k1 * k1 + k2 * k2;

    const complex_t u = u_hat[0 * stride_u + idx_u];
    const complex_t v = u_hat[1 * stride_u + idx_u];
    complex_t force1, force2;
    forcing(t, dt, u, v, 1, i_, j, &force1, &force2);
    complex_t L1_hat, L2_hat;
    incompressible_euler_2d_compute_L(k1,
                                      k2,
                                      absk2,
                                      stride_B,
                                      idx_B,
                                      B_hat.raw(),
                                      force1,
                                      force2,
                                      &L1_hat,
                                      &L2_hat);

    const real_t nu = visc.eval(zisa::sqrt(absk2));
    dudt_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + nu * u;
    dudt_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + nu * v;
  }
}

template <typename SpectralViscosity, typename Forcing>
__global__ void
incompressible_euler_3d_cuda_kernel(zisa::array_const_view<complex_t, 4> B_hat,
                                    zisa::array_const_view<complex_t, 4> u_hat,
                                    zisa::array_view<complex_t, 4> dudt_hat,
                                    SpectralViscosity visc,
                                    Forcing forcing,
                                    real_t t,
                                    real_t dt) {
  const unsigned i = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride_B = B_hat.shape(1) * B_hat.shape(2) * B_hat.shape(3);
  const unsigned i_B
      = i >= u_hat.shape(1) / 2 + 1 ? B_hat.shape(1) - u_hat.shape(1) + i : i;
  const unsigned j_B
      = j >= u_hat.shape(2) / 2 + 1 ? B_hat.shape(2) - u_hat.shape(2) + j : j;
  const unsigned idx_B
      = i_B * B_hat.shape(2) * B_hat.shape(3) + j_B * B_hat.shape(3) + k;
  const unsigned stride_u = u_hat.shape(1) * u_hat.shape(2) * u_hat.shape(3);
  const unsigned idx_u
      = i * u_hat.shape(2) * u_hat.shape(3) + j * u_hat.shape(3) + k;

  if (i < u_hat.shape(1) && j < u_hat.shape(2) && k < u_hat.shape(3)) {
    int i_ = i;
    int j_ = j;
    if (i_ >= u_hat.shape(1) / 2 + 1) {
      i_ -= u_hat.shape(1);
    }
    if (j_ >= u_hat.shape(2) / 2 + 1) {
      j_ -= u_hat.shape(2);
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t k3 = 2 * zisa::pi * k;
    const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;

    const complex_t u = u_hat[0 * stride_u + idx_u];
    const complex_t v = u_hat[1 * stride_u + idx_u];
    const complex_t w = u_hat[2 * stride_u + idx_u];
    complex_t force1, force2, force3;
    forcing(t, dt, u, v, w, 1, i_, j_, k, &force1, &force2, &force3);
    complex_t L1_hat, L2_hat, L3_hat;
    incompressible_euler_3d_compute_L(k1,
                                      k2,
                                      k3,
                                      absk2,
                                      stride_B,
                                      idx_B,
                                      B_hat.raw(),
                                      force1,
                                      force2,
                                      force3,
                                      &L1_hat,
                                      &L2_hat,
                                      &L3_hat);

    const real_t nu = visc.eval(zisa::sqrt(absk2));
    dudt_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + nu * u;
    dudt_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + nu * v;
    dudt_hat[2 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L3_hat + nu * w;
  }
}

template <typename SpectralViscosity, typename Forcing>
__global__ void incompressible_euler_2d_tracer_cuda_kernel(
    zisa::array_const_view<complex_t, 3> B_hat,
    zisa::array_const_view<complex_t, 3> u_hat,
    zisa::array_view<complex_t, 3> dudt_hat,
    SpectralViscosity visc,
    Forcing forcing,
    real_t t,
    real_t dt) {
  const unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride_B = B_hat.shape(1) * B_hat.shape(2);
  const unsigned i_B
      = i >= u_hat.shape(1) / 2 + 1 ? B_hat.shape(1) - u_hat.shape(1) + i : i;
  const unsigned idx_B = i_B * B_hat.shape(2) + j;
  const unsigned stride_u = u_hat.shape(1) * u_hat.shape(2);
  const unsigned idx_u = i * u_hat.shape(2) + j;

  if (i < u_hat.shape(1) && j < u_hat.shape(2)) {
    int i_ = i;
    if (i_ >= u_hat.shape(1) / 2 + 1) {
      i_ -= u_hat.shape(1);
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j;
    const real_t absk2 = k1 * k1 + k2 * k2;

    const complex_t u = u_hat[0 * stride_u + idx_u];
    const complex_t v = u_hat[1 * stride_u + idx_u];
    const complex_t rho = u_hat[2 * stride_u + idx_u];
    complex_t force1, force2;
    forcing(t, dt, u, v, rho, i_, j, &force1, &force2);
    complex_t L1_hat, L2_hat, L3_hat;
    incompressible_euler_2d_compute_L(k1,
                                      k2,
                                      absk2,
                                      stride_B,
                                      idx_B,
                                      B_hat.raw(),
                                      force1,
                                      force2,
                                      &L1_hat,
                                      &L2_hat);
    advection_2d(k1, k2, stride_B, idx_B, B_hat.raw() + 3 * stride_B, &L3_hat);

    const real_t nu = visc.eval(zisa::sqrt(absk2));
    dudt_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + nu * u;
    dudt_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + nu * v;
    dudt_hat[2 * stride_u + idx_u] = -L3_hat + nu * rho;
  }
}

template <typename SpectralViscosity, typename Forcing>
__global__ void incompressible_euler_3d_tracer_cuda_kernel(
    zisa::array_const_view<complex_t, 4> B_hat,
    zisa::array_const_view<complex_t, 4> u_hat,
    zisa::array_view<complex_t, 4> dudt_hat,
    SpectralViscosity visc,
    Forcing forcing,
    real_t t,
    real_t dt) {
  const unsigned i = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride_B = B_hat.shape(1) * B_hat.shape(2) * B_hat.shape(3);
  const unsigned i_B
      = i >= u_hat.shape(1) / 2 + 1 ? B_hat.shape(1) - u_hat.shape(1) + i : i;
  const unsigned j_B
      = j >= u_hat.shape(2) / 2 + 1 ? B_hat.shape(2) - u_hat.shape(2) + j : j;
  const unsigned idx_B
      = i_B * B_hat.shape(2) * B_hat.shape(3) + j_B * B_hat.shape(3) + k;
  const unsigned stride_u = u_hat.shape(1) * u_hat.shape(2) * u_hat.shape(3);
  const unsigned idx_u
      = i * u_hat.shape(2) * u_hat.shape(3) + j * u_hat.shape(3) + k;

  if (i < u_hat.shape(1) && j < u_hat.shape(2) && k < u_hat.shape(3)) {
    int i_ = i;
    int j_ = j;
    if (i_ >= u_hat.shape(1) / 2 + 1) {
      i_ -= u_hat.shape(1);
    }
    if (j_ >= u_hat.shape(2) / 2 + 1) {
      j_ -= u_hat.shape(2);
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t k3 = 2 * zisa::pi * k;
    const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;

    const complex_t u = u_hat[0 * stride_u + idx_u];
    const complex_t v = u_hat[1 * stride_u + idx_u];
    const complex_t w = u_hat[2 * stride_u + idx_u];
    const complex_t rho = u_hat[3 * stride_u + idx_u];
    complex_t force1, force2, force3;
    forcing(t, dt, u, v, w, rho, i_, j_, k, &force1, &force2, &force3);
    complex_t L1_hat, L2_hat, L3_hat, L4_hat;
    incompressible_euler_3d_compute_L(k1,
                                      k2,
                                      k3,
                                      absk2,
                                      stride_B,
                                      idx_B,
                                      B_hat.raw(),
                                      force1,
                                      force2,
                                      force3,
                                      &L1_hat,
                                      &L2_hat,
                                      &L3_hat);
    advection_3d(
        k1, k2, k3, stride_B, idx_B, B_hat.raw() + 6 * stride_B, &L4_hat);

    const real_t nu = visc.eval(zisa::sqrt(absk2));
    dudt_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + nu * u;
    dudt_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + nu * v;
    dudt_hat[2 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L3_hat + nu * w;
    dudt_hat[3 * stride_u + idx_u] = -L4_hat + nu * rho;
  }
}

template <int Dim>
void incompressible_euler_compute_B_cuda(
    const zisa::array_view<real_t, Dim + 1> &B,
    const zisa::array_const_view<real_t, Dim + 1> &u,
    const Grid<Dim> &grid) {}

template <>
void incompressible_euler_compute_B_cuda<2>(
    const zisa::array_view<real_t, 3> &B,
    const zisa::array_const_view<real_t, 3> &u,
    const Grid<2> &grid) {
  assert(B.memory_location() == zisa::device_type::cuda);
  assert(u.memory_location() == zisa::device_type::cuda);
  assert(B.shape(1) == u.shape(1));
  assert(B.shape(2) == u.shape(2));

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u.shape(2)), thread_dims.x),
      zisa::div_up(static_cast<int>(u.shape(1)), thread_dims.y),
      1);
  incompressible_euler_compute_B_cuda_kernel<2>
      <<<block_dims, thread_dims>>>(B, u, grid);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <>
void incompressible_euler_compute_B_cuda<3>(
    const zisa::array_view<real_t, 4> &B,
    const zisa::array_const_view<real_t, 4> &u,
    const Grid<3> &grid) {
  assert(B.memory_location() == zisa::device_type::cuda);
  assert(u.memory_location() == zisa::device_type::cuda);
  assert(B.shape(1) == u.shape(1));
  assert(B.shape(2) == u.shape(2));
  assert(B.shape(3) == u.shape(3));
  const dim3 thread_dims(32, 4, 4);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u.shape(3)), thread_dims.x),
      zisa::div_up(static_cast<int>(u.shape(2)), thread_dims.y),
      zisa::div_up(static_cast<int>(u.shape(1)), thread_dims.z));
  incompressible_euler_compute_B_cuda_kernel<3>
      <<<block_dims, thread_dims>>>(B, u, grid);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <int Dim>
void incompressible_euler_compute_B_tracer_cuda(
    const zisa::array_view<real_t, Dim + 1> &B,
    const zisa::array_const_view<real_t, Dim + 1> &u,
    const Grid<Dim> &grid) {}

template <>
void incompressible_euler_compute_B_tracer_cuda<2>(
    const zisa::array_view<real_t, 3> &B,
    const zisa::array_const_view<real_t, 3> &u,
    const Grid<2> &grid) {
  assert(B.memory_location() == zisa::device_type::cuda);
  assert(u.memory_location() == zisa::device_type::cuda);
  assert(B.shape(1) == u.shape(1));
  assert(B.shape(2) == u.shape(2));

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u.shape(2)), thread_dims.x),
      zisa::div_up(static_cast<int>(u.shape(1)), thread_dims.y),
      1);
  incompressible_euler_compute_B_tracer_cuda_kernel<2>
      <<<block_dims, thread_dims>>>(B, u, grid);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <>
void incompressible_euler_compute_B_tracer_cuda<3>(
    const zisa::array_view<real_t, 4> &B,
    const zisa::array_const_view<real_t, 4> &u,
    const Grid<3> &grid) {
  assert(B.memory_location() == zisa::device_type::cuda);
  assert(u.memory_location() == zisa::device_type::cuda);
  assert(B.shape(1) == u.shape(1));
  assert(B.shape(2) == u.shape(2));
  assert(B.shape(3) == u.shape(3));
  const dim3 thread_dims(32, 4, 4);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u.shape(3)), thread_dims.x),
      zisa::div_up(static_cast<int>(u.shape(2)), thread_dims.y),
      zisa::div_up(static_cast<int>(u.shape(1)), thread_dims.z));
  incompressible_euler_compute_B_tracer_cuda_kernel<3>
      <<<block_dims, thread_dims>>>(B, u, grid);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_2d_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    real_t t,
    real_t dt) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.y),
      1);

  incompressible_euler_2d_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, dudt_hat, visc, forcing, t, dt);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_3d_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    real_t t,
    real_t dt) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);
  const dim3 thread_dims(32, 4, 4);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(3)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.z));
  incompressible_euler_3d_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, dudt_hat, visc, forcing, t, dt);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_2d_tracer_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    real_t t,
    real_t dt) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.y),
      1);

  incompressible_euler_2d_tracer_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, dudt_hat, visc, forcing, t, dt);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_3d_tracer_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    real_t t,
    real_t dt) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);
  const dim3 thread_dims(32, 4, 4);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
      zisa::div_up(static_cast<int>(u_hat.shape(3)), thread_dims.z));
  incompressible_euler_3d_tracer_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, dudt_hat, visc, forcing, t, dt);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

}

#endif
