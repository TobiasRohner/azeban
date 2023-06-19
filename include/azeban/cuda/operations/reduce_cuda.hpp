#ifndef AZEBAN_CUDA_OPERATIONS_REDUCE_CUDA_HPP_
#define AZEBAN_CUDA_OPERATIONS_REDUCE_CUDA_HPP_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

real_t reduce_sum_cuda(const zisa::array_const_view<real_t, 1> &data);

}

#endif
