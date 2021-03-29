#include <azeban/cuda/operations/scale_cuda.hpp>

#include <azeban/cuda/operations/scale_cuda_impl.cuh>

namespace azeban {

#define AZEBAN_INSTANCIATE_SCALE_CUDA(TYPE)                                    \
  template void scale_cuda<TYPE>(const TYPE &,                                 \
                                 const zisa::array_view<TYPE, 1> &);

AZEBAN_INSTANCIATE_SCALE_CUDA(real_t)
AZEBAN_INSTANCIATE_SCALE_CUDA(complex_t)

#undef AZEBAN_INSTANCIATE_SCALE_CUDA

}
