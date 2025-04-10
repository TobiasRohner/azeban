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

#include <azeban/cuda/operations/clamp_cuda.hpp>
#include <azeban/cuda/operations/clamp_cuda_impl.cuh>

namespace azeban {

#define AZEBAN_INSTANCIATE_CLAMP_CUDA(TYPE)                                    \
  template void clamp_cuda<TYPE>(const zisa::array_view<TYPE, 1> &, real_t);

AZEBAN_INSTANCIATE_CLAMP_CUDA(real_t)
AZEBAN_INSTANCIATE_CLAMP_CUDA(complex_t)

#undef AZEBAN_INSTANCIATE_CLAMP_CUDA

}
