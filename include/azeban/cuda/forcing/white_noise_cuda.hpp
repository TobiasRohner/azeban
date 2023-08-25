#ifndef WHITE_NOISE_CUDA_HPP_
#define WHITE_NOISE_CUDA_HPP_

#include <azeban/config.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename RNG>
void white_noise_pre_cuda(const zisa::array_view<real_t, 2> &pot, RNG *rng);
template <typename RNG>
void white_noise_pre_cuda(const zisa::array_view<real_t, 3> &pot, RNG *rng);

#define AZEBAN_INSTANTIATE_WHITE_NOISE_PRE_CUDA(RNG)                           \
  extern template void white_noise_pre_cuda<RNG>(                              \
      const zisa::array_view<real_t, 2> &, RNG *);                             \
  extern template void white_noise_pre_cuda<RNG>(                              \
      const zisa::array_view<real_t, 3> &, RNG *);

AZEBAN_INSTANTIATE_WHITE_NOISE_PRE_CUDA(curandStateMRG32k3a_t)
AZEBAN_INSTANTIATE_WHITE_NOISE_PRE_CUDA(curandStateXORWOW_t)

#undef AZEBAN_INSTANTIATE_WHITE_NOISE_PRE_CUDA

}

#endif
