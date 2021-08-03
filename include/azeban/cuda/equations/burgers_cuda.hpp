#ifndef BURGERS_CUDA_H_
#define BURGERS_CUDS_H_

#include <azeban/config.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename SpectralViscosity>
void burgers_cuda(const zisa::array_view<complex_t, 2> &u,
                  const zisa::array_const_view<complex_t, 2> &u_squared,
                  const SpectralViscosity &visc);

#define AZEBAN_INSTANTIATE_BURGERS_CUDA(TYPE)                                  \
  extern template void burgers_cuda<TYPE>(                                     \
      const zisa::array_view<complex_t, 2> &,                                  \
      const zisa::array_const_view<complex_t, 2> &,                            \
      const TYPE &);

AZEBAN_INSTANTIATE_BURGERS_CUDA(Step1D)
AZEBAN_INSTANTIATE_BURGERS_CUDA(SmoothCutoff1D)
AZEBAN_INSTANTIATE_BURGERS_CUDA(Quadratic)
AZEBAN_INSTANTIATE_BURGERS_CUDA(NoViscosity)

#undef AZEBAN_INSTANTIATE_BURGERS_CUDA

}

#endif
