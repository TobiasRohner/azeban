#ifndef WHITE_NOISE_HIGH_FREQ_CUDA_IMPL_H_
#define WHITE_NOISE_HIGH_FREQ_CUDA_IMPL_H_

#include <azeban/cuda/forcing/white_noise_high_freq_cuda.hpp>
#include <curand_kernel.h>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename RNG>
__device__ __inline__ float normal(RNG *rng, float) {
  return curand_normal(rng);
}

template <typename RNG>
__device__ __inline__ double normal(RNG *rng, double) {
  return curand_normal_double(rng);
}

template <typename RNG>
__device__ __inline__ real_t normal(RNG *rng) {
  return normal(rng, real_t{});
}

template <typename RNG>
__global__ void
white_noise_high_freq_pre_cuda_kernel(zisa::array_view<real_t, 2> pot,
                                      RNG *rng) {
  const unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= pot.shape(0) || j >= pot.shape(1)) {
    return;
  }

  const unsigned id = i * pot.shape(1) + j;
  RNG local_state = rng[id];
  pot(i, j) = normal(&local_state);
  rng[id] = local_state;
}

template <typename RNG>
void white_noise_high_freq_pre_cuda(const zisa::array_view<real_t, 2> &pot,
                                    RNG *rng) {
  assert(pot.memory_location() == zisa::device_type::cuda);

  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(pot.shape(1)), thread_dims.x),
      zisa::div_up(static_cast<int>(pot.shape(0)), thread_dims.y),
      1);
  white_noise_high_freq_pre_cuda_kernel<<<block_dims, thread_dims>>>(pot, rng);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

template <typename RNG>
void white_noise_high_freq_pre_cuda(const zisa::array_view<real_t, 3> &,
                                    RNG *) {
  // Nothing to do here
}

}

#endif
