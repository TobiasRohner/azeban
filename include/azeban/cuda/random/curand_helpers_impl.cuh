#ifndef AZEBAN_CUDA_RANDOM_CURAND_HELPERS_IMPL_HPP_
#define AZEBAN_CUDA_RANDOM_CURAND_HELPERS_IMPL_HPP_

#include "curand_helpers.hpp"
#include <curand_kernel.h>
#include <azeban/cuda/cuda_check_error.hpp>
#include <zisa/math/basic_functions.hpp>


namespace azeban {

template<typename RNG>
__global__ void curand_init_state_kernel(typename RNGTraits<RNG>::state_t state, size_t N, unsigned long long seed) {
  const unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < N) {
    curand_init(seed, id, 0, &state[id]);
  }
}

template<typename RNG, typename>
void curand_allocate_state(typename RNGTraits<RNG>::state_t &state, size_t N, unsigned long long seed) {
  using state_t = typename RNGTraits<RNG>::state_t;
  const auto err = cudaMalloc(&state, N * sizeof(state_t));
  cudaCheckError(err);
  const dim3 thread_dims(1024, 1, 1);
  const dim3 block_dims(zisa::div_up(static_cast<int>(N), thread_dims.x), 1, 1);
  curand_init_state_kernel<RNG><<<block_dims, thread_dims>>>(state, N, seed);
  ZISA_CHECK_CUDA_DEBUG;
  cudaDeviceSynchronize();
}

template<typename RNG, typename>
void curand_free_state(typename RNGTraits<RNG>::state_t &state) {
  cudaFree(state);
}

}


#endif
