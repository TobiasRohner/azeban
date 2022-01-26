#include <azeban/cuda/random/curand_helpers.hpp>
#include <azeban/cuda/random/curand_helpers_impl.cuh>

namespace azeban {

#define AZEBAN_INSTANTIATE_CURAND_HELPERS(RNG)                                 \
  template void curand_allocate_state<RNG>(                                    \
      typename RNGTraits<RNG>::state_t **, size_t, unsigned long long);        \
  template void curand_free_state<RNG>(typename RNGTraits<RNG>::state_t *);

AZEBAN_INSTANTIATE_CURAND_HELPERS(curandStateMRG32k3a_t);
AZEBAN_INSTANTIATE_CURAND_HELPERS(curandStateXORWOW_t);

#undef AZEBAN_INSTANTIATE_CURAND_HELPERS

}
