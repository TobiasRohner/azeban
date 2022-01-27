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
#ifndef COPY_PADDED_CUDA_H_
#define COPY_PADDED_CUDA_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

void copy_to_padded_cuda(const zisa::array_view<complex_t, 1> &,
                         const zisa::array_const_view<complex_t, 1> &,
                         bool pad_x,
                         const complex_t &);
void copy_to_padded_cuda(const zisa::array_view<complex_t, 1> &,
                         const zisa::array_const_view<complex_t, 1> &,
                         const complex_t &);
void copy_to_padded_cuda(const zisa::array_view<complex_t, 2> &,
                         const zisa::array_const_view<complex_t, 2> &,
                         bool pad_x,
                         bool pad_y,
                         const complex_t &);
void copy_to_padded_cuda(const zisa::array_view<complex_t, 2> &,
                         const zisa::array_const_view<complex_t, 2> &,
                         const complex_t &);
void copy_to_padded_cuda(const zisa::array_view<complex_t, 3> &,
                         const zisa::array_const_view<complex_t, 3> &,
                         bool pad_x,
                         bool pad_y,
                         bool pad_z,
                         const complex_t &);
void copy_to_padded_cuda(const zisa::array_view<complex_t, 3> &,
                         const zisa::array_const_view<complex_t, 3> &,
                         const complex_t &);

}

#endif
