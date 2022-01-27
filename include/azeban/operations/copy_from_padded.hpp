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
#ifndef AZEBAN_OPERATIONS_COPY_FROM_PADDED_HPP_
#define AZEBAN_OPERATIONS_COPY_FROM_PADDED_HPP_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

void copy_from_padded(bool pad_x,
                      int compact_dim,
                      const zisa::array_view<complex_t, 1> &dst,
                      const zisa::array_const_view<complex_t, 1> &src);
void copy_from_padded(const zisa::array_view<complex_t, 1> &dst,
                      const zisa::array_const_view<complex_t, 1> &src);
void copy_from_padded(bool pad_x,
                      bool pad_y,
                      int compact_dim,
                      const zisa::array_view<complex_t, 2> &dst,
                      const zisa::array_const_view<complex_t, 2> &src);
void copy_from_padded(const zisa::array_view<complex_t, 2> &dst,
                      const zisa::array_const_view<complex_t, 2> &src);
void copy_from_padded(bool pad_x,
                      bool pad_y,
                      bool pad_z,
                      int compact_dim,
                      const zisa::array_view<complex_t, 3> &dst,
                      const zisa::array_const_view<complex_t, 3> &src);
void copy_from_padded(const zisa::array_view<complex_t, 3> &dst,
                      const zisa::array_const_view<complex_t, 3> &src);

}

#endif
