#ifndef SCALE_CUDA_H_
#define SCALE_CUDA_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename Scalar>
void scale_cuda(const Scalar &a, const zisa::array_view<Scalar, 1> &x);

#define AZEBAN_INSTANCIATE_SCALE_CUDA(TYPE)                                    \
  extern template void scale_cuda<TYPE>(const TYPE &,                          \
                                        const zisa::array_view<TYPE, 1> &);

AZEBAN_INSTANCIATE_SCALE_CUDA(real_t)
AZEBAN_INSTANCIATE_SCALE_CUDA(complex_t)

#undef AZEBAN_INSTANCIATE_SCALE_CUDA

}

#endif
