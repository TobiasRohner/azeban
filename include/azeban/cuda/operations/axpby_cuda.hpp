#ifndef AXPBY_CUDA_H_
#define AXPBY_CUDA_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename Scalar>
void axpby_cuda(const Scalar &a,
                const zisa::array_const_view<Scalar, 1> &x,
                const Scalar &b,
                const zisa::array_view<Scalar, 1> &y);

#define AZEBAN_INSTANCIATE_AXPBY_CUDA(TYPE)                                    \
  extern template void axpby_cuda<TYPE>(                                       \
      const TYPE &,                                                            \
      const zisa::array_const_view<TYPE, 1> &,                                 \
      const TYPE &,                                                            \
      const zisa::array_view<TYPE, 1> &);

AZEBAN_INSTANCIATE_AXPBY_CUDA(real_t)
AZEBAN_INSTANCIATE_AXPBY_CUDA(complex_t)

#undef AZEBAN_INSTANCIATE_AXPBY_CUDA

}

#endif
