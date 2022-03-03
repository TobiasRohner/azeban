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
#ifndef AZEBAN_OPERATIONS_TRANSPOSE_HPP_
#define AZEBAN_OPERATIONS_TRANSPOSE_HPP_

#if AZEBAN_HAS_MPI

#include <azeban/config.hpp>
#include <azeban/memory/workspace.hpp>
#include <memory>
#include <mpi.h>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
class Transpose {
public:
  Transpose(MPI_Comm comm,
            const zisa::array_const_view<complex_t, Dim + 1> &from,
            const zisa::array_view<complex_t, Dim + 1> &to);
  Transpose(const Transpose &) = default;
  Transpose(Transpose &&) = default;
  Transpose &operator=(const Transpose &) = default;
  Transpose &operator=(Transpose &&) = default;

  zisa::device_type location() const;

  zisa::shape_t<Dim + 2> buffer_shape() const;
  void set_send_buffer(const zisa::array_view<complex_t, Dim + 2> &sendbuf);
  void set_recv_buffer(const zisa::array_view<complex_t, Dim + 2> &recvbuf);

  void eval();

private:
  MPI_Comm comm_;
  int size_, rank_;
  zisa::device_type location_;
  zisa::array_const_view<complex_t, Dim + 1> from_;
  zisa::array_view<complex_t, Dim + 1> to_;
  zisa::array_view<complex_t, Dim + 2> sendbuf_;
  zisa::array_view<complex_t, Dim + 2> recvbuf_;
  std::unique_ptr<zisa::shape_t<Dim + 1>[]> from_shapes_;
  std::unique_ptr<zisa::shape_t<Dim + 1>[]> to_shapes_;
  zisa::shape_t<Dim + 1> max_from_size_;
  zisa::shape_t<Dim + 1> max_to_size_;

  void preprocess_cpu();
  void preprocess();
  void communicate();
  void postprocess_cpu();
  void postprocess();
};

void transpose(const zisa::array_view<complex_t, 3> &dst,
               const zisa::array_const_view<complex_t, 3> &src,
               MPI_Comm comm);
void transpose(const zisa::array_view<complex_t, 4> &dst,
               const zisa::array_const_view<complex_t, 4> &src,
               MPI_Comm comm);

}

#endif

#endif
