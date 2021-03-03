#ifndef INCOMPRESSIBLE_EULER_CUDA_IMPL_H_
#define INCOMPRESSIBLE_EULER_CUDA_IMPL_H_

#include "incompressible_euler_cuda.hpp"

namespace azeban {

template <int Dim>
__global__ void incompressible_euler_compute_B_cuda_kernel(
    zisa::array_view<real_t, Dim + 1> B,
    zisa::array_const_view<real_t, Dim + 1> u,
    zisa::int_t N_phys,
    zisa::int_t N_phys_pad) {}

template <>
__global__ void incompressible_euler_compute_B_cuda_kernel<2>(
    zisa::array_view<real_t, 3> B,
    zisa::array_const_view<real_t, 3> u,
    zisa::int_t N_phys,
    zisa::int_t N_phys_pad) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned stride = u.shape(1) * u.shape(2);
  const unsigned idx = i * u.shape(1) + j;

  const real_t norm = 1.0 / (N_phys * N_phys * N_phys_pad * N_phys_pad);
  if (i < u.shape(1) && j < u.shape(2)) {
    const real_t u1 = u[0 * stride + idx];
    const real_t u2 = u[1 * stride + idx];
    B[0 * stride + idx] = norm * u1 * u1;
    B[1 * stride + idx] = norm * u1 * u2;
    B[2 * stride + idx] = norm * u2 * u1;
    B[3 * stride + idx] = norm * u2 * u2;
  }
}

template <>
__global__ void incompressible_euler_compute_B_cuda_kernel<3>(
    zisa::array_view<real_t, 4> B,
    zisa::array_const_view<real_t, 4> u,
    zisa::int_t N_phys,
    zisa::int_t N_phys_pad) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned k = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned stride = u.shape(1) * u.shape(2) * u.shape(3);
  const unsigned idx = i * u.shape(1) * u.shape(2) + j * u.shape(1) + k;

  const real_t norm
      = 1.0 / (N_phys * N_phys * N_phys * N_phys_pad * N_phys_pad * N_phys_pad);
  if (i < u.shape(1) && j < u.shape(2) && k < u.shape(3)) {
    const real_t u1 = u[0 * stride + idx];
    const real_t u2 = u[1 * stride + idx];
    const real_t u3 = u[2 * stride + idx];
    B[0 * stride + idx] = norm * u1 * u1;
    B[1 * stride + idx] = norm * u1 * u2;
    B[2 * stride + idx] = norm * u1 * u3;
    B[3 * stride + idx] = norm * u2 * u1;
    B[4 * stride + idx] = norm * u2 * u2;
    B[5 * stride + idx] = norm * u2 * u3;
    B[6 * stride + idx] = norm * u3 * u1;
    B[7 * stride + idx] = norm * u3 * u2;
    B[8 * stride + idx] = norm * u3 * u3;
  }
}

template <typename SpectralViscosity>
__global__ void
incompressible_euler_2d_cuda_kernel(zisa::array_const_view<complex_t, 3> B_hat,
                                    zisa::array_view<complex_t, 3> u_hat,
                                    SpectralViscosity visc) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned stride = u_hat.shape(1) * u_hat.shape(2);
  const unsigned idx = i * u_hat.shape(1) + j;

  if (i < u_hat.shape(1) && j < u_hat.shape(2)) {
    int i_ = i;
    if (i_ >= u_hat.shape(1) / 2 + 1) {
      i_ -= u_hat.shape(1);
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j;

    const complex_t b1_hat = complex_t(0, k1) * B_hat[0 * stride + idx]
                             + complex_t(0, k2) * B_hat[1 * stride + idx];
    const complex_t b2_hat = complex_t(0, k1) * B_hat[2 * stride + idx]
                             + complex_t(0, k2) * B_hat[3 * stride + idx];

    const real_t absk2 = k1 * k1 + k2 * k2;
    const complex_t L1_hat
        = (1. - (k1 * k1) / absk2) * b1_hat + (0. - (k1 * k2) / absk2) * b2_hat;
    const complex_t L2_hat
        = (0. - (k2 * k1) / absk2) * b1_hat + (1. - (k2 * k2) / absk2) * b2_hat;

    const real_t v = visc.eval(zisa::sqrt(absk2));
    u_hat[0 * stride + idx] = -L1_hat + v * u_hat[0 * stride + idx];
    u_hat[1 * stride + idx] = -L2_hat + v * u_hat[1 * stride + idx];
  }
}

template <typename SpectralViscosity>
__global__ void
incompressible_euler_3d_cuda_kernel(zisa::array_const_view<complex_t, 4> B_hat,
                                    zisa::array_view<complex_t, 4> u_hat,
                                    SpectralViscosity visc) {
  // TODO: Implement
}

template <int Dim>
void incompressible_euler_compute_B_cuda(
    const zisa::array_view<real_t, Dim + 1> &B,
    const zisa::array_const_view<real_t, Dim + 1> &u,
    zisa::int_t N_phys,
    zisa::int_t N_phys_pad) {}

template <>
void incompressible_euler_compute_B_cuda<2>(
    const zisa::array_view<real_t, 3> &B,
    const zisa::array_const_view<real_t, 3> &u,
    zisa::int_t N_phys,
    zisa::int_t N_phys_pad) {
  assert(B.memory_location() == zisa::device_type::cuda);
  assert(u.memory_location() == zisa::device_type::cuda);
  assert(B.shape(1) == u.shape(1));
  assert(B.shape(2) == u.shape(2));
  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::min(zisa::div_up(static_cast<int>(u.shape(1)), thread_dims.x), 32),
      zisa::min(zisa::div_up(static_cast<int>(u.shape(2)), thread_dims.y), 32),
      1);
  incompressible_euler_compute_B_cuda_kernel<2>
      <<<block_dims, thread_dims>>>(B, u, N_phys, N_phys_pad);
}

template <>
void incompressible_euler_compute_B_cuda<3>(
    const zisa::array_view<real_t, 4> &B,
    const zisa::array_const_view<real_t, 4> &u,
    zisa::int_t N_phys,
    zisa::int_t N_phys_pad) {
  assert(B.memory_location() == zisa::device_type::cuda);
  assert(u.memory_location() == zisa::device_type::cuda);
  assert(B.shape(1) == u.shape(1));
  assert(B.shape(2) == u.shape(2));
  assert(B.shape(3) == u.shape(3));
  const dim3 thread_dims(8, 8, 8);
  const dim3 block_dims(
      zisa::min(zisa::div_up(static_cast<int>(u.shape(1)), thread_dims.x), 8),
      zisa::min(zisa::div_up(static_cast<int>(u.shape(2)), thread_dims.y), 8),
      zisa::min(zisa::div_up(static_cast<int>(u.shape(3)), thread_dims.z), 8));
  incompressible_euler_compute_B_cuda_kernel<3>
      <<<block_dims, thread_dims>>>(B, u, N_phys, N_phys_pad);
}

template <typename SpectralViscosity>
void incompressible_euler_2d_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_view<complex_t, 3> &u_hat,
    const SpectralViscosity &visc) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);
  assert(B_hat.shape(1) == u_hat.shape(1));
  assert(B_hat.shape(2) == u_hat.shape(2));
  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::min(zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
                32),
      zisa::min(zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
                32),
      1);
  incompressible_euler_2d_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, visc);
}

template <typename SpectralViscosity>
void incompressible_euler_3d_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_view<complex_t, 4> &u_hat,
    const SpectralViscosity &visc) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);
  assert(B_hat.shape(1) == u_hat.shape(1));
  assert(B_hat.shape(2) == u_hat.shape(2));
  assert(B_hat.shape(3) == u_hat.shape(3));
  const dim3 thread_dims(8, 8, 8);
  const dim3 block_dims(
      zisa::min(zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
                8),
      zisa::min(zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
                8),
      zisa::min(zisa::div_up(static_cast<int>(u_hat.shape(3)), thread_dims.z),
                8));
  incompressible_euler_3d_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, visc);
}

}

#endif
