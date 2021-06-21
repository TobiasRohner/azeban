#include <azeban/config.hpp>
#include <azeban/cuda/operations/leray_cuda.hpp>

namespace azeban {

__global__ void leray_cuda_kernel(zisa::array_view<complex_t, 3> u_hat) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int N_phys = u_hat.shape(1);
  const int N_fourier = N_phys / 2 + 1;

  if (i < u_hat.shape(1) && j < u_hat.shape(2)) {
    int i_ = i;
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
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;
  const int N_phys = u_hat.shape(1);
  const int N_fourier = N_phys / 2 + 1;

  if (i < u_hat.shape(1) && j < u_hat.shape(2) && k < u_hat.shape(3)) {
    int i_ = i;
    if (i_ >= N_fourier) {
      i_ -= N_phys;
    }
    int j_ = j;
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
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
      1);
  leray_cuda_kernel<<<block_dims, thread_dims>>>(u_hat);
  ZISA_CHECK_CUDA_DEBUG;
  cudaDeviceSynchronize();
}

void leray_cuda(const zisa::array_view<complex_t, 4> &u_hat) {
  const dim3 thread_dims(4, 4, 32);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u_hat.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(u_hat.shape(2)), thread_dims.y),
      zisa::div_up(static_cast<int>(u_hat.shape(3)), thread_dims.z));
  leray_cuda_kernel<<<block_dims, thread_dims>>>(u_hat);
  ZISA_CHECK_CUDA_DEBUG;
  cudaDeviceSynchronize();
}

}
