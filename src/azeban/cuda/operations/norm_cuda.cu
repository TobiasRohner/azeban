/*
 * This file is part of azeban (https://github.com/TobiasRohner/azeban).
 * Copyright (c) 2021 Tobias Rohner.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
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
                                  real_t p);				       \
  template real_t max_norm_cuda<TYPE>(const zisa::array_const_view<TYPE, 1> &);

AZEBAN_INSTANTIATE_REDUCE_CUDA(real_t)
AZEBAN_INSTANTIATE_REDUCE_CUDA(complex_t)

#undef AZEBAN_INSTANTIATE_REDUCE_CUDA

}
