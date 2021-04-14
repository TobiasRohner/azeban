#ifndef INCOMPRESSIBLE_EULER_CUDA_IMPL_H_
#define INCOMPRESSIBLE_EULER_CUDA_IMPL_H_

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
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned stride = u.shape(1) * u.shape(2);
  const unsigned idx = i * u.shape(2) + j;

  const real_t norm
      = 1.0 / (zisa::pow<2>(grid.N_phys) * zisa::pow<2>(grid.N_phys_pad));
  if (i < u.shape(1) && j < u.shape(2)) {
    const real_t u1 = u[0 * stride + idx];
    const real_t u2 = u[1 * stride + idx];
    incompressible_euler_2d_compute_B(stride, idx, norm, u1, u2, B.raw());
  }
}

template <>
__global__ void incompressible_euler_compute_B_cuda_kernel<3>(
    zisa::array_view<real_t, 4> B,
    zisa::array_const_view<real_t, 4> u,
    Grid<3> grid) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned k = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned stride = u.shape(1) * u.shape(2) * u.shape(3);
  const unsigned idx = i * u.shape(2) * u.shape(3) + j * u.shape(3) + k;

  const real_t norm
      = 1.0 / (zisa::pow<3>(grid.N_phys) * zisa::pow<3>(grid.N_phys_pad));
  if (i < u.shape(1) && j < u.shape(2) && k < u.shape(3)) {
    const real_t u1 = u[0 * stride + idx];
    const real_t u2 = u[1 * stride + idx];
    const real_t u3 = u[2 * stride + idx];
    incompressible_euler_3d_compute_B(stride, idx, norm, u1, u2, u3, B.raw());
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
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned stride = u.shape(1) * u.shape(2);
  const unsigned idx = i * u.shape(2) + j;

  const real_t norm
      = 1.0 / (zisa::pow<2>(grid.N_phys) * zisa::pow<2>(grid.N_phys_pad));
  if (i < u.shape(1) && j < u.shape(2)) {
    const real_t u1 = u[0 * stride + idx];
    const real_t u2 = u[1 * stride + idx];
    const real_t rho = u[2 * stride + idx];
    incompressible_euler_2d_compute_B(stride, idx, norm, u1, u2, B.raw());
    advection_2d_compute_B(
        stride, idx, norm, rho, u1, u2, B.raw() + 3 * stride);
  }
}

template <>
__global__ void incompressible_euler_compute_B_tracer_cuda_kernel<3>(
    zisa::array_view<real_t, 4> B,
    zisa::array_const_view<real_t, 4> u,
    Grid<3> grid) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned k = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned stride = u.shape(1) * u.shape(2) * u.shape(3);
  const unsigned idx = i * u.shape(2) * u.shape(3) + j * u.shape(3) + k;

  const real_t norm
      = 1.0 / (zisa::pow<3>(grid.N_phys) * zisa::pow<3>(grid.N_phys_pad));
  if (i < u.shape(1) && j < u.shape(2) && k < u.shape(3)) {
    const real_t u1 = u[0 * stride + idx];
    const real_t u2 = u[1 * stride + idx];
    const real_t u3 = u[2 * stride + idx];
    const real_t rho = u[3 * stride + idx];
    incompressible_euler_3d_compute_B(stride, idx, norm, u1, u2, u3, B.raw());
    advection_3d_compute_B(
        stride, idx, norm, rho, u1, u2, u3, B.raw() + 6 * stride);
  }
}

template <typename SpectralViscosity>
__global__ void
incompressible_euler_2d_cuda_kernel(zisa::array_const_view<complex_t, 3> B_hat,
                                    zisa::array_view<complex_t, 3> u_hat,
                                    SpectralViscosity visc) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
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

    complex_t L1_hat, L2_hat;
    incompressible_euler_2d_compute_L(
        k1, k2, absk2, stride_B, idx_B, B_hat.raw(), &L1_hat, &L2_hat);

    const real_t v = visc.eval(zisa::sqrt(absk2));
    u_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + v * u_hat[0 * stride_u + idx_u];
    u_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + v * u_hat[1 * stride_u + idx_u];
  }
}

template <typename SpectralViscosity>
__global__ void
incompressible_euler_3d_cuda_kernel(zisa::array_const_view<complex_t, 4> B_hat,
                                    zisa::array_view<complex_t, 4> u_hat,
                                    SpectralViscosity visc) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned k = blockIdx.z * blockDim.z + threadIdx.z;
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

    complex_t L1_hat, L2_hat, L3_hat;
    incompressible_euler_3d_compute_L(k1,
                                      k2,
                                      k3,
                                      absk2,
                                      stride_B,
                                      idx_B,
                                      B_hat.raw(),
                                      &L1_hat,
                                      &L2_hat,
                                      &L3_hat);

    const real_t v = visc.eval(zisa::sqrt(absk2));
    u_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + v * u_hat[0 * stride_u + idx_u];
    u_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + v * u_hat[1 * stride_u + idx_u];
    u_hat[2 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L3_hat + v * u_hat[2 * stride_u + idx_u];
  }
}

template <typename SpectralViscosity>
__global__ void incompressible_euler_2d_tracer_cuda_kernel(
    zisa::array_const_view<complex_t, 3> B_hat,
    zisa::array_view<complex_t, 3> u_hat,
    SpectralViscosity visc) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
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

    complex_t L1_hat, L2_hat, L3_hat;
    incompressible_euler_2d_compute_L(
        k1, k2, absk2, stride_B, idx_B, B_hat.raw(), &L1_hat, &L2_hat);
    advection_2d(k1, k2, stride_B, idx_B, B_hat.raw() + 3 * stride_B, &L3_hat);

    const real_t v = visc.eval(zisa::sqrt(absk2));
    u_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + v * u_hat[0 * stride_u + idx_u];
    u_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + v * u_hat[1 * stride_u + idx_u];
    u_hat[2 * stride_u + idx_u] = -L3_hat + v * u_hat[2 * stride_u + idx_u];
  }
}

template <typename SpectralViscosity>
__global__ void incompressible_euler_3d_tracer_cuda_kernel(
    zisa::array_const_view<complex_t, 4> B_hat,
    zisa::array_view<complex_t, 4> u_hat,
    SpectralViscosity visc) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned k = blockIdx.z * blockDim.z + threadIdx.z;
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

    complex_t L1_hat, L2_hat, L3_hat, L4_hat;
    incompressible_euler_3d_compute_L(k1,
                                      k2,
                                      k3,
                                      absk2,
                                      stride_B,
                                      idx_B,
                                      B_hat.raw(),
                                      &L1_hat,
                                      &L2_hat,
                                      &L3_hat);
    advection_3d(
        k1, k2, k3, stride_B, idx_B, B_hat.raw() + 6 * stride_B, &L4_hat);

    const real_t v = visc.eval(zisa::sqrt(absk2));
    u_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + v * u_hat[0 * stride_u + idx_u];
    u_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + v * u_hat[1 * stride_u + idx_u];
    u_hat[2 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L3_hat + v * u_hat[2 * stride_u + idx_u];
    u_hat[3 * stride_u + idx_u] = -L4_hat + v * u_hat[3 * stride_u + idx_u];
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
      zisa::div_up(static_cast<int>(u.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u.shape(2)), thread_dims.y),
      1);
  incompressible_euler_compute_B_cuda_kernel<2>
      <<<block_dims, thread_dims>>>(B, u, grid);
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
  const dim3 thread_dims(4, 4, 32);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u.shape(2)), thread_dims.y),
      zisa::div_up(static_cast<int>(u.shape(3)), thread_dims.z));
  incompressible_euler_compute_B_cuda_kernel<3>
      <<<block_dims, thread_dims>>>(B, u, grid);
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
      zisa::div_up(static_cast<int>(u.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u.shape(2)), thread_dims.y),
      1);
  incompressible_euler_compute_B_tracer_cuda_kernel<2>
      <<<block_dims, thread_dims>>>(B, u, grid);
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
  const dim3 thread_dims(4, 4, 32);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u.shape(2)), thread_dims.y),
      zisa::div_up(static_cast<int>(u.shape(3)), thread_dims.z));
  incompressible_euler_compute_B_tracer_cuda_kernel<3>
      <<<block_dims, thread_dims>>>(B, u, grid);
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity>
void incompressible_euler_2d_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_view<complex_t, 3> &u_hat,
    const SpectralViscosity &visc) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
      1);

  incompressible_euler_2d_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, visc);
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity>
void incompressible_euler_3d_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_view<complex_t, 4> &u_hat,
    const SpectralViscosity &visc) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);
  const dim3 thread_dims(4, 4, 32);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
      zisa::div_up(static_cast<int>(u_hat.shape(3)), thread_dims.z));
  incompressible_euler_3d_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, visc);
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity>
void incompressible_euler_2d_tracer_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_view<complex_t, 3> &u_hat,
    const SpectralViscosity &visc) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
      1);

  incompressible_euler_2d_tracer_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, visc);
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity>
void incompressible_euler_3d_tracer_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_view<complex_t, 4> &u_hat,
    const SpectralViscosity &visc) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);
  const dim3 thread_dims(4, 4, 32);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
      zisa::div_up(static_cast<int>(u_hat.shape(3)), thread_dims.z));
  incompressible_euler_3d_tracer_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, visc);
  ZISA_CHECK_CUDA_DEBUG;
}

}

#endif
