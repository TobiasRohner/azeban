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
#ifndef NORM_CUDA_H_
#define NORM_CUDA_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename Scalar>
real_t norm_cuda(const zisa::array_const_view<Scalar, 1> &data, real_t p);

#define AZEBAN_INSTANTIATE_REDUCE_CUDA(TYPE)                                   \
  extern template real_t norm_cuda<TYPE>(                                      \
      const zisa::array_const_view<TYPE, 1> &, real_t p);

AZEBAN_INSTANTIATE_REDUCE_CUDA(real_t)
AZEBAN_INSTANTIATE_REDUCE_CUDA(complex_t)

#undef AZEBAN_INSTANTIATE_REDUCE_CUDA

}

#endif
