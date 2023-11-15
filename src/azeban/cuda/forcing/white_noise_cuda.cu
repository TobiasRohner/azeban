#include <azeban/cuda/forcing/white_noise_cuda.hpp>
#include <azeban/cuda/forcing/white_noise_cuda_impl.cuh>
#include <curand.h>
#include <curand_kernel.h>

namespace azeban {

#define AZEBAN_INSTANTIATE_WHITE_NOISE_PRE_CUDA(RNG)                           \
  template void white_noise_pre_cuda<RNG>(const zisa::array_view<real_t, 2> &, \
                                          RNG *);                              \
  template void white_noise_pre_cuda<RNG>(const zisa::array_view<real_t, 3> &, \
                                          RNG *);

AZEBAN_INSTANTIATE_WHITE_NOISE_PRE_CUDA(curandStateMRG32k3a_t)
AZEBAN_INSTANTIATE_WHITE_NOISE_PRE_CUDA(curandStateXORWOW_t)

#undef AZEBAN_INSTANTIATE_WHITE_NOISE_PRE_CUDA

}
