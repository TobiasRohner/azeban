#ifndef AZEBAN_CUDA_RANDOM_CURAND_HELPERS_HPP_
#define AZEBAN_CUDA_RANDOM_CURAND_HELPERS_HPP_

#include <azeban/random/rng_traits.hpp>

namespace azeban {

template <typename RNG,
          typename = typename std::enable_if<RNGTraits<RNG>::location
                                             == zisa::device_type::cuda>::type>
void curand_allocate_state(typename RNGTraits<RNG>::state_t **state,
                           size_t N,
                           unsigned long long seed);

template <typename RNG,
          typename = typename std::enable_if<RNGTraits<RNG>::location
                                             == zisa::device_type::cuda>::type>
void curand_free_state(typename RNGTraits<RNG>::state_t *state);

#define AZEBAN_INSTANTIATE_CURAND_HELPERS(RNG)                                 \
  extern template void curand_allocate_state<RNG>(                             \
      typename RNGTraits<RNG>::state_t **, size_t, unsigned long long);        \
  extern template void curand_free_state<RNG>(                                 \
      typename RNGTraits<RNG>::state_t *);

AZEBAN_INSTANTIATE_CURAND_HELPERS(curandStateMRG32k3a_t);
AZEBAN_INSTANTIATE_CURAND_HELPERS(curandStateXORWOW_t);

#undef AZEBAN_INSTANTIATE_CURAND_HELPERS

}

#endif
