#ifndef AZEBAN_CUDA_OPERATIONS_REDUCE_CUDA_HPP_
#define AZEBAN_CUDA_OPERATIONS_REDUCE_CUDA_HPP_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

float reduce_sum_cuda(const zisa::array_const_view<float, 1> &data);
double reduce_sum_cuda(const zisa::array_const_view<double, 1> &data);

}

#endif
