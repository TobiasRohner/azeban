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
    real_t t,
    real_t dt,
    unsigned long i_base,
    unsigned long j_base,
    zisa::shape_t<3> shape_phys) {
  const unsigned long i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long j = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long stride_B = B_hat.shape(1) * B_hat.shape(2);
  const unsigned long idx_B = i * B_hat.shape(2) + j;
  const unsigned long stride_u = u_hat.shape(1) * u_hat.shape(2);
  const unsigned long idx_u = i * u_hat.shape(2) + j;

  if (i < u_hat.shape(1) && j < u_hat.shape(2)) {
    long i_ = i_base + i;
    if (i_ >= zisa::integer_cast<long>(shape_phys[1] / 2 + 1)) {
      i_ -= shape_phys[1];
    }
    long j_ = j_base + j;
    if (j_ >= zisa::integer_cast<long>(shape_phys[2] / 2 + 1)) {
      j_ -= shape_phys[2];
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t absk2 = k1 * k1 + k2 * k2;

    const complex_t u = u_hat[0 * stride_u + idx_u];
    const complex_t v = u_hat[1 * stride_u + idx_u];
    complex_t force1, force2;
    forcing(t, dt, u, v, 1, j_, i_, &force1, &force2);
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

    const real_t nu = visc.eval(zisa::sqrt(absk2));
    dudt_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + nu * u;
    dudt_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + nu * v;
  }
}

template <typename SpectralViscosity, typename Forcing>
__global__ void incompressible_euler_mpi_3d_cuda_kernel(
    zisa::array_const_view<complex_t, 4> B_hat,
    zisa::array_const_view<complex_t, 4> u_hat,
    zisa::array_view<complex_t, 4> dudt_hat,
    SpectralViscosity visc,
    Forcing forcing,
    real_t t,
    real_t dt,
    unsigned long i_base,
    unsigned long j_base,
    unsigned long k_base,
    zisa::shape_t<4> shape_phys) {
  const unsigned long i = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long stride_B
      = B_hat.shape(1) * B_hat.shape(2) * B_hat.shape(3);
  const unsigned long idx_B
      = i * B_hat.shape(2) * B_hat.shape(3) + j * B_hat.shape(3) + k;
  const unsigned long stride_u
      = u_hat.shape(1) * u_hat.shape(2) * u_hat.shape(3);
  const unsigned long idx_u
      = i * u_hat.shape(2) * u_hat.shape(3) + j * u_hat.shape(3) + k;

  if (i < u_hat.shape(1) && j < u_hat.shape(2) && k < u_hat.shape(3)) {
    long i_ = i_base + i;
    if (i_ >= zisa::integer_cast<long>(shape_phys[1] / 2 + 1)) {
      i_ -= shape_phys[1];
    }
    long j_ = j_base + j;
    if (j_ >= zisa::integer_cast<long>(shape_phys[2] / 2 + 1)) {
      j_ -= shape_phys[2];
    }
    long k_ = k_base + k;
    if (k_ >= zisa::integer_cast<long>(shape_phys[3] / 2 + 1)) {
      k_ -= shape_phys[3];
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t k3 = 2 * zisa::pi * k_;
    const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;

    const complex_t u = u_hat[0 * stride_u + idx_u];
    const complex_t v = u_hat[1 * stride_u + idx_u];
    const complex_t w = u_hat[2 * stride_u + idx_u];
    complex_t force1, force2, force3;
    forcing(t, dt, u, v, w, 1, k_, j_, i_, &force1, &force2, &force3);
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
__global__ void incompressible_euler_mpi_2d_tracer_cuda_kernel(
    zisa::array_const_view<complex_t, 3> B_hat,
    zisa::array_const_view<complex_t, 3> u_hat,
    zisa::array_view<complex_t, 3> dudt_hat,
    SpectralViscosity visc,
    Forcing forcing,
    real_t t,
    real_t dt,
    unsigned long i_base,
    unsigned long j_base,
    zisa::shape_t<3> shape_phys) {
  const unsigned long i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long j = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long stride_B = B_hat.shape(1) * B_hat.shape(2);
  const unsigned long idx_B = i * B_hat.shape(2) + j;
  const unsigned long stride_u = u_hat.shape(1) * u_hat.shape(2);
  const unsigned long idx_u = i * u_hat.shape(2) + j;

  if (i < u_hat.shape(1) && j < u_hat.shape(2)) {
    long i_ = i_base + i;
    if (i_ >= zisa::integer_cast<long>(shape_phys[1] / 2 + 1)) {
      i_ -= shape_phys[1];
    }
    long j_ = j_base + j;
    if (j_ >= zisa::integer_cast<long>(shape_phys[2] / 2 + 1)) {
      j_ -= shape_phys[2];
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t absk2 = k1 * k1 + k2 * k2;

    const complex_t u = u_hat[0 * stride_u + idx_u];
    const complex_t v = u_hat[1 * stride_u + idx_u];
    const complex_t rho = u_hat[2 * stride_u + idx_u];
    complex_t force1, force2;
    forcing(t, dt, u, v, rho, j_, i_, &force1, &force2);
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

    const real_t nu = visc.eval(zisa::sqrt(absk2));
    dudt_hat[0 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L1_hat + nu * u;
    dudt_hat[1 * stride_u + idx_u]
        = absk2 == 0 ? 0 : -L2_hat + nu * v;
    dudt_hat[2 * stride_u + idx_u] = -L3_hat + nu * rho;
  }
}

template <typename SpectralViscosity, typename Forcing>
__global__ void incompressible_euler_mpi_3d_tracer_cuda_kernel(
    zisa::array_const_view<complex_t, 4> B_hat,
    zisa::array_const_view<complex_t, 4> u_hat,
    zisa::array_view<complex_t, 4> dudt_hat,
    SpectralViscosity visc,
    Forcing forcing,
    real_t t,
    real_t dt,
    unsigned long i_base,
    unsigned long j_base,
    unsigned long k_base,
    zisa::shape_t<4> shape_phys) {
  const unsigned long i = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long k = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long stride_B
      = B_hat.shape(1) * B_hat.shape(2) * B_hat.shape(3);
  const unsigned long idx_B
      = i * B_hat.shape(2) * B_hat.shape(3) + j * B_hat.shape(3) + k;
  const unsigned long stride_u
      = u_hat.shape(1) * u_hat.shape(2) * u_hat.shape(3);
  const unsigned long idx_u
      = i * u_hat.shape(2) * u_hat.shape(3) + j * u_hat.shape(3) + k;

  if (i < u_hat.shape(1) && j < u_hat.shape(2) && k < u_hat.shape(3)) {
    long i_ = i_base + i;
    if (i_ >= zisa::integer_cast<long>(shape_phys[1] / 2 + 1)) {
      i_ -= shape_phys[1];
    }
    long j_ = j_base + j;
    if (j_ >= zisa::integer_cast<long>(shape_phys[2] / 2 + 1)) {
      j_ -= shape_phys[2];
    }
    long k_ = k_base + k;
    if (k_ >= zisa::integer_cast<long>(shape_phys[3] / 2 + 1)) {
      k_ -= shape_phys[3];
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j_;
    const real_t k3 = 2 * zisa::pi * k_;
    const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;

    const complex_t u = u_hat[0 * stride_u + idx_u];
    const complex_t v = u_hat[1 * stride_u + idx_u];
    const complex_t w = u_hat[2 * stride_u + idx_u];
    const complex_t rho = u_hat[3 * stride_u + idx_u];
    complex_t force1, force2, force3;
    forcing(t, dt, u, v, w, rho, k_, j_, i_, &force1, &force2, &force3);
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

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_mpi_2d_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    real_t t,
    real_t dt,
    unsigned long i_base,
    unsigned long j_base,
    const zisa::shape_t<3> &shape_phys) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(u_hat.shape(2),
                   zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(u_hat.shape(1),
                   zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      1);

  incompressible_euler_mpi_2d_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, dudt_hat, visc, forcing, t, dt, i_base, j_base, shape_phys);
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
    real_t t,
    real_t dt,
    unsigned long i_base,
    unsigned long j_base,
    unsigned long k_base,
    const zisa::shape_t<4> &shape_phys) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);
  const dim3 thread_dims(32, 4, 4);
  const dim3 block_dims(
      zisa::div_up(u_hat.shape(3),
                   zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(u_hat.shape(2),
                   zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      zisa::div_up(u_hat.shape(1),
                   zisa::integer_cast<zisa::int_t>(thread_dims.z)));
  incompressible_euler_mpi_3d_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat,
      u_hat,
      dudt_hat,
      visc,
      forcing,
      t,
      dt,
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
    real_t t,
    real_t dt,
    unsigned long i_base,
    unsigned long j_base,
    const zisa::shape_t<3> &shape_phys) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(u_hat.shape(2),
                   zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(u_hat.shape(1),
                   zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      1);

  incompressible_euler_mpi_2d_tracer_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat, u_hat, dudt_hat, visc, forcing, t, dt, i_base, j_base, shape_phys);
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
    real_t t,
    real_t dt,
    unsigned long i_base,
    unsigned long j_base,
    unsigned long k_base,
    const zisa::shape_t<4> &shape_phys) {
  assert(B_hat.memory_location() == zisa::device_type::cuda);
  assert(u_hat.memory_location() == zisa::device_type::cuda);
  const dim3 thread_dims(32, 4, 4);
  const dim3 block_dims(
      zisa::div_up(u_hat.shape(3),
                   zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(u_hat.shape(2),
                   zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      zisa::div_up(u_hat.shape(1),
                   zisa::integer_cast<zisa::int_t>(thread_dims.z)));
  incompressible_euler_mpi_3d_tracer_cuda_kernel<<<block_dims, thread_dims>>>(
      B_hat,
      u_hat,
      dudt_hat,
      visc,
      forcing,
      t,
      dt,
      i_base,
      j_base,
      k_base,
      shape_phys);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

}

#endif
