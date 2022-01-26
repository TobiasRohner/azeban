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
#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#if AZEBAN_HAS_MPI

#include <azeban/config.hpp>
#include <mpi.h>
#include <zisa/memory/array_view.hpp>

namespace azeban {

void transpose(const zisa::array_view<complex_t, 3> &dst,
               const zisa::array_const_view<complex_t, 3> &src,
               MPI_Comm comm);
void transpose(const zisa::array_view<complex_t, 4> &dst,
               const zisa::array_const_view<complex_t, 4> &src,
               MPI_Comm comm);

}

#endif

#endif
