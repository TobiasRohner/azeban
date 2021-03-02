#include <azeban/config.hpp>
#include <azeban/copy_padded.hpp>
#include <azeban/cuda/copy_padded_cuda_impl.cuh>

namespace azeban {

#define AZEBAN_INSTANTIATE_COPY_PADDED_CUDA(TYPE)                              \
  template void copy_to_padded_cuda<TYPE>(                                     \
      const zisa::array_view<TYPE, 1> &,                                       \
      const zisa::array_const_view<TYPE, 1> &,                                 \
      const TYPE &);                                                           \
  template void copy_to_padded_cuda<TYPE>(                                     \
      const zisa::array_view<TYPE, 2> &,                                       \
      const zisa::array_const_view<TYPE, 2> &,                                 \
      const TYPE &);                                                           \
  template void copy_to_padded_cuda<TYPE>(                                     \
      const zisa::array_view<TYPE, 3> &,                                       \
      const zisa::array_const_view<TYPE, 3> &,                                 \
      const TYPE &);                                                           \
  template void copy_from_padded_cuda<TYPE>(                                   \
      const zisa::array_view<TYPE, 1> &,                                       \
      const zisa::array_const_view<TYPE, 1> &);                                \
  template void copy_from_padded_cuda<TYPE>(                                   \
      const zisa::array_view<TYPE, 2> &,                                       \
      const zisa::array_const_view<TYPE, 2> &);                                \
  template void copy_from_padded_cuda<TYPE>(                                   \
      const zisa::array_view<TYPE, 3> &,                                       \
      const zisa::array_const_view<TYPE, 3> &);

AZEBAN_INSTANTIATE_COPY_PADDED_CUDA(real_t)
AZEBAN_INSTANTIATE_COPY_PADDED_CUDA(complex_t)

#undef AZEBAN_INSTANTIATE_COPY_PADDED_CUDA

}
