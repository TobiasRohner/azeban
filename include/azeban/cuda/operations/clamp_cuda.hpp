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
#ifndef CLAMP_CUDA_H_
#define CLAMP_CUDA_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename Scalar>
void clamp_cuda(const zisa::array_view<Scalar, 1> &x, real_t a);

#define AZEBAN_INSTANCIATE_CLAMP_CUDA(TYPE)                                    \
  extern template void clamp_cuda<TYPE>(const zisa::array_view<TYPE, 1> &,     \
                                        real_t);

AZEBAN_INSTANCIATE_CLAMP_CUDA(real_t)
AZEBAN_INSTANCIATE_CLAMP_CUDA(complex_t)

#undef AZEBAN_INSTANCIATE_CLAMP_CUDA

}

#endif
