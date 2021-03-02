#include <azeban/cuda/operations/axpy_cuda.hpp>

#include <azeban/cuda/operations/axpy_cuda_impl.cuh>

namespace azeban {

#define AZEBAN_INSTANCIATE_AXPY_CUDA(TYPE)                                     \
  template void axpy_cuda<TYPE>(const TYPE &,                                  \
                                const zisa::array_const_view<TYPE, 1> &,       \
                                const zisa::array_view<TYPE, 1> &);

AZEBAN_INSTANCIATE_AXPY_CUDA(real_t)
AZEBAN_INSTANCIATE_AXPY_CUDA(complex_t)

#undef AZEBAN_INSTANCIATE_AXPY_CUDA

}
