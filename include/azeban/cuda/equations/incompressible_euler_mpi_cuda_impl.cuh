#ifndef AZEBAN_CUDA_EQUATIONS_INCOMPRESSIBLE_EULER_MPI_CUDA_IMPL_HPP_
#define AZEBAN_CUDA_EQUATIONS_INCOMPRESSIBLE_EULER_MPI_CUDA_IMPL_HPP_

#include "incompressible_euler_mpi_cuda.hpp"
#include <azeban/equations/advection_functions.hpp>
#include <azeban/equations/incompressible_euler_functions.hpp>

namespace azeban {

template <typename SpectralViscosity, typename Forcing>
__global__ void incompressible_euler_mpi_2d_cuda_kernel(
    zisa::array_const_view<complex_t, 3> B_hat,
    zisa::array_const_view<complex_t, 3> u_hat,
    zisa::array_view<complex_t, 3> dudt_hat,
    SpectralViscosity visc,
    Forcing forcing,
    unsigned i_base,
    unsigned j_base,
    zisa::shape_t<3> shape_phys) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned stride_B = B_hat.shape(1) * B_hat.shape(2);
  const unsigned idx_B = i * B_hat.shape(2) + j;
  const unsigned stride_u = u_hat.shape(1) * u_hat.shape(2);
  const unsigned idx_u = i * u_hat.shape(2) + j;

  if (i < u_hat.shape(1) && j < u_hat.shape(2)) {
    int i_ = i_base + i;
    if (i_ >= zisa::integer_cast<int>(shape_phys[1] / 2 + 1)) {
      i_ -= shape_phys[1];
    }
    int j_ = j_base + j;
    if (j_ >= zisa::integer_cast<int>(shape_phys[2] / 2 + 1)) {
      j_ -= shape_phys[2];
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t absk2 = k1 * k1 + k2 * k2;

    complex_t force1, force2;
    forcing(0, k2, k1, &force1, &force2);
    complex_t L1_hat, L2_hat;
    incompressible_euler_2d_compute_L(k2,
                                      k1,
                                      absk2,
                                      stride_B,
                                      idx_B,
                                      B_hat.raw(),
                                      force1,
                                      force2,
                                      &L1_hat,
                                      &L2_hat);

    const real_t v = visc.eval(zisa::sqrt(absk2));
    dudt_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + v * u_hat[0 * stride_u + idx_u];
    dudt_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + v * u_hat[1 * stride_u + idx_u];
  }
}

template <typename SpectralViscosity, typename Forcing>
__global__ void incompressible_euler_mpi_3d_cuda_kernel(
    zisa::array_const_view<complex_t, 4> B_hat,
    zisa::array_const_view<complex_t, 4> u_hat,
    zisa::array_view<complex_t, 4> dudt_hat,
    SpectralViscosity visc,
    Forcing forcing,
    unsigned i_base,
    unsigned j_base,
    unsigned k_base,
    zisa::shape_t<4> shape_phys) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned k = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned stride_B = B_hat.shape(1) * B_hat.shape(2) * B_hat.shape(3);
  const unsigned idx_B
      = i * B_hat.shape(2) * B_hat.shape(3) + j * B_hat.shape(3) + k;
  const unsigned stride_u = u_hat.shape(1) * u_hat.shape(2) * u_hat.shape(3);
  const unsigned idx_u
      = i * u_hat.shape(2) * u_hat.shape(3) + j * u_hat.shape(3) + k;

  if (i < u_hat.shape(1) && j < u_hat.shape(2) && k < u_hat.shape(3)) {
    int i_ = i_base + i;
    if (i_ >= zisa::integer_cast<int>(shape_phys[1] / 2 + 1)) {
      i_ -= shape_phys[1];
    }
    int j_ = j_base + j;
    if (j_ >= zisa::integer_cast<int>(shape_phys[2] / 2 + 1)) {
      j_ -= shape_phys[2];
    }
    int k_ = k_base + k;
    if (k_ >= zisa::integer_cast<int>(shape_phys[3] / 2 + 1)) {
      k_ -= shape_phys[3];
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t k3 = 2 * zisa::pi * k_;
    const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;

    complex_t force1, force2, force3;
    forcing(0, k3, k2, k1, &force1, &force2, &force3);
    complex_t L1_hat, L2_hat, L3_hat;
    incompressible_euler_3d_compute_L(k3,
                                      k2,
                                      k1,
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

    const real_t v = visc.eval(zisa::sqrt(absk2));
    dudt_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + v * u_hat[0 * stride_u + idx_u];
    dudt_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + v * u_hat[1 * stride_u + idx_u];
    dudt_hat[2 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L3_hat + v * u_hat[2 * stride_u + idx_u];
  }
}

template <typename SpectralViscosity, typename Forcing>
__global__ void incompressible_euler_mpi_2d_tracer_cuda_kernel(
    zisa::array_const_view<complex_t, 3> B_hat,
    zisa::array_const_view<complex_t, 3> u_hat,
    zisa::array_view<complex_t, 3> dudt_hat,
    SpectralViscosity visc,
    Forcing forcing,
    unsigned i_base,
    unsigned j_base,
    zisa::shape_t<3> shape_phys) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned stride_B = B_hat.shape(1) * B_hat.shape(2);
  const unsigned idx_B = i * B_hat.shape(2) + j;
  const unsigned stride_u = u_hat.shape(1) * u_hat.shape(2);
  const unsigned idx_u = i * u_hat.shape(2) + j;

  if (i < u_hat.shape(1) && j < u_hat.shape(2)) {
    int i_ = i_base + i;
    if (i_ >= zisa::integer_cast<int>(shape_phys[1] / 2 + 1)) {
      i_ -= shape_phys[1];
    }
    int j_ = j_base + j;
    if (j_ >= zisa::integer_cast<int>(shape_phys[2] / 2 + 1)) {
      j_ -= shape_phys[2];
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t absk2 = k1 * k1 + k2 * k2;

    complex_t force1, force2;
    forcing(0, k2, k1, &force1, &force2);
    complex_t L1_hat, L2_hat, L3_hat;
    incompressible_euler_2d_compute_L(k2,
                                      k1,
                                      absk2,
                                      stride_B,
                                      idx_B,
                                      B_hat.raw(),
                                      force1,
                                      force2,
                                      &L1_hat,
                                      &L2_hat);
    advection_2d(k2, k1, stride_B, idx_B, B_hat.raw() + 3 * stride_B, &L3_hat);

    const real_t v = visc.eval(zisa::sqrt(absk2));
    dudt_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + v * u_hat[0 * stride_u + idx_u];
    dudt_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + v * u_hat[1 * stride_u + idx_u];
    dudt_hat[2 * stride_u + idx_u] = -L3_hat + v * u_hat[2 * stride_u + idx_u];
  }
}

template <typename SpectralViscosity, typename Forcing>
__global__ void incompressible_euler_mpi_3d_tracer_cuda_kernel(
    zisa::array_const_view<complex_t, 4> B_hat,
    zisa::array_const_view<complex_t, 4> u_hat,
    zisa::array_view<complex_t, 4> dudt_hat,
    SpectralViscosity visc,
    Forcing forcing,
    unsigned i_base,
    unsigned j_base,
    unsigned k_base,
    zisa::shape_t<4> shape_phys) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned k = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned stride_B = B_hat.shape(1) * B_hat.shape(2) * B_hat.shape(3);
  const unsigned idx_B
      = i * B_hat.shape(2) * B_hat.shape(3) + j * B_hat.shape(3) + k;
  const unsigned stride_u = u_hat.shape(1) * u_hat.shape(2) * u_hat.shape(3);
  const unsigned idx_u
      = i * u_hat.shape(2) * u_hat.shape(3) + j * u_hat.shape(3) + k;

  if (i < u_hat.shape(1) && j < u_hat.shape(2) && k < u_hat.shape(3)) {
    int i_ = i_base + i;
    if (i_ >= zisa::integer_cast<int>(shape_phys[1] / 2 + 1)) {
      i_ -= shape_phys[1];
    }
    int j_ = j_base + j;
    if (j_ >= zisa::integer_cast<int>(shape_phys[2] / 2 + 1)) {
      j_ -= shape_phys[2];
    }
    int k_ = k_base + k;
    if (k_ >= zisa::integer_cast<int>(shape_phys[3] / 2 + 1)) {
      k_ -= shape_phys[3];
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t k3 = 2 * zisa::pi * k_;
    const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;

    complex_t force1, force2, force3;
    forcing(0, k3, k2, k1, &force1, &force2, &force3);
    complex_t L1_hat, L2_hat, L3_hat, L4_hat;
    incompressible_euler_3d_compute_L(k3,
                                      k2,
                                      k1,
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
        k3, k2, k1, stride_B, idx_B, B_hat.raw() + 6 * stride_B, &L4_hat);

    const real_t v = visc.eval(zisa::sqrt(absk2));
    dudt_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + v * u_hat[0 * stride_u + idx_u];
    dudt_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + v * u_hat[1 * stride_u + idx_u];
    dudt_hat[2 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L3_hat + v * u_hat[2 * stride_u + idx_u];
    dudt_hat[3 * stride_u + idx_u] = -L4_hat + v * u_hat[3 * stride_u + idx_u];
  }
}

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_mpi_2d_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    unsigned i_base,
    unsigned j_base,
    const zisa::shape_t<3> &shape_phys) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
      1);

  incompressible_euler_mpi_2d_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, dudt_hat, visc, forcing, i_base, j_base, shape_phys);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_mpi_3d_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    unsigned i_base,
    unsigned j_base,
    unsigned k_base,
    const zisa::shape_t<4> &shape_phys) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);
  const dim3 thread_dims(4, 4, 32);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
      zisa::div_up(static_cast<int>(u_hat.shape(3)), thread_dims.z));
  incompressible_euler_mpi_3d_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat,
      u_hat,
      dudt_hat,
      visc,
      forcing,
      i_base,
      j_base,
      k_base,
      shape_phys);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_mpi_2d_tracer_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    unsigned i_base,
    unsigned j_base,
    const zisa::shape_t<3> &shape_phys) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
      1);

  incompressible_euler_mpi_2d_tracer_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, dudt_hat, visc, forcing, i_base, j_base, shape_phys);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_mpi_3d_tracer_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    unsigned i_base,
    unsigned j_base,
    unsigned k_base,
    const zisa::shape_t<4> &shape_phys) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);
  const dim3 thread_dims(4, 4, 32);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
      zisa::div_up(static_cast<int>(u_hat.shape(3)), thread_dims.z));
  incompressible_euler_mpi_3d_tracer_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat,
      u_hat,
      dudt_hat,
      visc,
      forcing,
      i_base,
      j_base,
      k_base,
      shape_phys);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

}

#endif
