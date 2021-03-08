#include <azeban/cuda/operations/axpby_cuda.hpp>

#include <azeban/cuda/operations/axpby_cuda_impl.cuh>

namespace azeban {

#define AZEBAN_INSTANCIATE_AXPBY_CUDA(TYPE)                                    \
  template void axpby_cuda<TYPE>(const TYPE &,                                 \
                                 const zisa::array_const_view<TYPE, 1> &,      \
                                 const TYPE &,                                 \
                                 const zisa::array_view<TYPE, 1> &);

AZEBAN_INSTANCIATE_AXPBY_CUDA(real_t)
AZEBAN_INSTANCIATE_AXPBY_CUDA(complex_t)

#undef AZEBAN_INSTANCIATE_AXPBY_CUDA

}
