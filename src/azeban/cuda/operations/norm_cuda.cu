#include <azeban/cuda/operations/norm_cuda.hpp>
#include <azeban/cuda/operations/norm_cuda_impl.cuh>
#include <zisa/config.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/comparison.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

#define AZEBAN_INSTANTIATE_REDUCE_CUDA(TYPE)                                   \
  template real_t norm_cuda<TYPE>(const zisa::array_const_view<TYPE, 1> &,     \
                                  real_t p);

AZEBAN_INSTANTIATE_REDUCE_CUDA(real_t)
AZEBAN_INSTANTIATE_REDUCE_CUDA(complex_t)

#undef AZEBAN_INSTANTIATE_REDUCE_CUDA

}
