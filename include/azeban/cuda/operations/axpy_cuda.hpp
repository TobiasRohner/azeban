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
#ifndef AXPY_CUDA_H_
#define AXPY_CUDA_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename Scalar>
void axpy_cuda(const Scalar &a,
               const zisa::array_const_view<Scalar, 1> &x,
               const zisa::array_view<Scalar, 1> &y);

#define AZEBAN_INSTANCIATE_AXPY_CUDA(TYPE)                                     \
  extern template void axpy_cuda<TYPE>(                                        \
      const TYPE &,                                                            \
      const zisa::array_const_view<TYPE, 1> &,                                 \
      const zisa::array_view<TYPE, 1> &);

AZEBAN_INSTANCIATE_AXPY_CUDA(real_t)
AZEBAN_INSTANCIATE_AXPY_CUDA(complex_t)

#undef AZEBAN_INSTANCIATE_AXPY_CUDA

}

#endif
