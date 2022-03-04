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
#if AZEBAN_HAS_MPI

#include <azeban/mpi_types.hpp>
#include <azeban/operations/transpose.hpp>
#include <azeban/profiler.hpp>
#include <iostream>
#include <memory>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/utils/logging.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/operations/transpose.hpp>
#endif

namespace azeban {

template <int Dim>
Transpose<Dim>::Transpose(
    MPI_Comm comm,
    const zisa::array_const_view<complex_t, Dim + 1> &from,
    const zisa::array_view<complex_t, Dim + 1> &to)
    : Transpose(comm, from.shape(), to.shape(), from.memory_location()) {
  set_from_array(from);
  set_to_array(to);
}

template <int Dim>
Transpose<Dim>::Transpose(MPI_Comm comm,
                          const zisa::shape_t<Dim + 1> &from_shape,
                          const zisa::shape_t<Dim + 1> &to_shape,
                          zisa::device_type location)
    : comm_(comm),
      location_(location),
      from_(from_shape, nullptr),
      to_(to_shape, nullptr),
      sendbuf_({}, nullptr),
      recvbuf_({}, nullptr) {
  MPI_Comm_size(comm_, &size_);
  MPI_Comm_rank(comm_, &rank_);
  from_shapes_ = std::make_unique<zisa::shape_t<Dim + 1>[]>(size_);
  to_shapes_ = std::make_unique<zisa::shape_t<Dim + 1>[]>(size_);
  MPI_Allgather(&from_shape,
                1,
                mpi_type(from_shape),
                from_shapes_.get(),
                1,
                mpi_type(from_shape),
                comm_);
  MPI_Allgather(&to_shape,
                1,
                mpi_type(to_shape),
                to_shapes_.get(),
                1,
                mpi_type(to_shape),
                comm_);
  for (int d = 0; d <= Dim; ++d) {
    max_from_size_[d] = 0;
    max_to_size_[d] = 0;
    for (int r = 0; r < size_; ++r) {
      if (max_from_size_[d] < from_shapes_[r][d]) {
        max_from_size_[d] = from_shapes_[r][d];
      }
      if (max_to_size_[d] < to_shapes_[r][d]) {
        max_to_size_[d] = to_shapes_[r][d];
      }
    }
  }
}

template <int Dim>
zisa::device_type Transpose<Dim>::location() const {
  return location_;
}

template <>
zisa::shape_t<4> Transpose<2>::buffer_shape() const {
  return {size_, max_from_size_[0], max_to_size_[1], max_from_size_[1]};
}

template <>
zisa::shape_t<5> Transpose<3>::buffer_shape() const {
  return {size_,
          max_from_size_[0],
          max_to_size_[1],
          max_from_size_[2],
          max_from_size_[1]};
}

template <int Dim>
void Transpose<Dim>::set_send_buffer(
    const zisa::array_view<complex_t, Dim + 2> &sendbuf) {
  LOG_ERR_IF(sendbuf.memory_location() != location(),
             "Send buffer is in the wrong memory location");
  LOG_ERR_IF(sendbuf.shape() != buffer_shape(),
             "Send buffer has the wrong shape");
  sendbuf_ = sendbuf;
}

template <int Dim>
void Transpose<Dim>::set_recv_buffer(
    const zisa::array_view<complex_t, Dim + 2> &recvbuf) {
  LOG_ERR_IF(recvbuf.memory_location() != location(),
             "Receive buffer is in the wrong memory location");
  LOG_ERR_IF(recvbuf.shape() != buffer_shape(),
             "Receive buffer has the wrong shape");
  recvbuf_ = recvbuf;
}

template <int Dim>
void Transpose<Dim>::set_from_array(
    const zisa::array_const_view<complex_t, Dim + 1> &from) {
  LOG_ERR_IF(from.memory_location() != location_, "Wrong memory_location");
  LOG_ERR_IF(from.shape() != from_shapes_[rank_], "Wrong array shape");
  from_ = from;
}

template <int Dim>
void Transpose<Dim>::set_to_array(
    const zisa::array_view<complex_t, Dim + 1> &to) {
  LOG_ERR_IF(to.memory_location() != location_, "Wrong memory_location");
  LOG_ERR_IF(to.shape() != to_shapes_[rank_], "Wrong array shape");
  to_ = to;
}

template <int Dim>
void Transpose<Dim>::eval() {
  AZEBAN_PROFILE_START("Transpose::eval");
  LOG_ERR_IF(sendbuf_.raw() == nullptr, "Send buffer uninitialized");
  LOG_ERR_IF(recvbuf_.raw() == nullptr, "Receive buffer uninitialized");
  LOG_ERR_IF(sendbuf_.raw() == recvbuf_.raw(),
             "In-place operation is not supported");
  preprocess();
  communicate();
  postprocess();
  AZEBAN_PROFILE_STOP("Transpose::eval");
}

template <>
void Transpose<2>::preprocess_cpu() {
  static constexpr zisa::int_t BLOCKSIZE = 16;
  zisa::int_t j_offset = 0;
  for (int r = 0; r < size_; ++r) {
    for (zisa::int_t d = 0; d < from_.shape(0); ++d) {
      for (zisa::int_t ib = 0; ib < from_.shape(1); ib += BLOCKSIZE) {
        for (zisa::int_t jb = 0; jb < to_shapes_[r][1]; jb += BLOCKSIZE) {
          const zisa::int_t i_end = zisa::min(ib + BLOCKSIZE, from_.shape(1));
          const zisa::int_t j_end = zisa::min(jb + BLOCKSIZE, to_shapes_[r][1]);
          for (zisa::int_t i = ib; i < i_end; ++i) {
            for (zisa::int_t j = jb; j < j_end; ++j) {
              sendbuf_(r, d, j, i) = from_(d, i, j + j_offset);
            }
          }
        }
      }
    }
    j_offset += to_shapes_[r][1];
  }
}

template <>
void Transpose<3>::preprocess_cpu() {
  static constexpr zisa::int_t BLOCKSIZE = 8;
  zisa::int_t k_offset = 0;
  for (int r = 0; r < size_; ++r) {
    for (zisa::int_t d = 0; d < from_.shape(0); ++d) {
      for (zisa::int_t ib = 0; ib < from_.shape(1); ib += BLOCKSIZE) {
        for (zisa::int_t jb = 0; jb < from_.shape(2); jb += BLOCKSIZE) {
          for (zisa::int_t kb = 0; kb < to_shapes_[r][1]; kb += BLOCKSIZE) {
            const zisa::int_t i_end = zisa::min(ib + BLOCKSIZE, from_.shape(1));
            const zisa::int_t j_end = zisa::min(jb + BLOCKSIZE, from_.shape(2));
            const zisa::int_t k_end
                = zisa::min(kb + BLOCKSIZE, to_shapes_[r][1]);
            for (zisa::int_t i = ib; i < i_end; ++i) {
              for (zisa::int_t j = jb; j < j_end; ++j) {
                for (zisa::int_t k = kb; k < k_end; ++k) {
                  sendbuf_(r, d, k, j, i) = from_(d, i, j, k + k_offset);
                }
              }
            }
          }
        }
      }
    }
    k_offset += to_shapes_[r][1];
  }
}

template <int Dim>
void Transpose<Dim>::preprocess() {
  AZEBAN_PROFILE_START("Transpose::preprocess");
  if (location_ == zisa::device_type::cpu) {
    preprocess_cpu();
  }
#if ZISA_HAS_CUDA
  else if (location_ == zisa::device_type::cuda) {
    transpose_cuda_preprocess(
        from_, sendbuf_, from_shapes_.get(), to_shapes_.get(), rank_);
  }
#endif
  AZEBAN_PROFILE_STOP("Transpose::preprocess");
}

template <int Dim>
void Transpose<Dim>::communicate() {
  AZEBAN_PROFILE_START("Transpose::communicate");
  // TODO: Use GPUDirect
  if (location_ == zisa::device_type::cpu) {
    MPI_Alltoall(sendbuf_.raw(),
                 sendbuf_.size() / size_,
                 mpi_type<complex_t>(),
                 recvbuf_.raw(),
                 recvbuf_.size() / size_,
                 mpi_type<complex_t>(),
                 comm_);
  }
#if ZISA_HAS_CUDA
  else if (location_ == zisa::device_type::cuda) {
    zisa::array<complex_t, Dim + 2> sendbuf(sendbuf_.shape(),
                                            zisa::device_type::cpu);
    zisa::array<complex_t, Dim + 2> recvbuf(recvbuf_.shape(),
                                            zisa::device_type::cpu);
    zisa::copy(sendbuf, sendbuf_);
    MPI_Alltoall(sendbuf.raw(),
                 sendbuf.size() / size_,
                 mpi_type<complex_t>(),
                 recvbuf.raw(),
                 recvbuf.size() / size_,
                 mpi_type<complex_t>(),
                 comm_);
    zisa::copy(recvbuf_, recvbuf);
  }
#endif
  else {
    LOG_ERR("Unsupported Device Type");
  }
  AZEBAN_PROFILE_STOP("Transpose::communicate");
}

template <>
void Transpose<2>::postprocess_cpu() {
  zisa::int_t j_offset = 0;
  for (int r = 0; r < size_; ++r) {
    for (zisa::int_t d = 0; d < to_.shape(0); ++d) {
      for (zisa::int_t i = 0; i < to_.shape(1); ++i) {
        for (zisa::int_t j = 0; j < from_shapes_[r][1]; ++j) {
          to_(d, i, j + j_offset) = recvbuf_(r, d, i, j);
        }
      }
    }
    j_offset += from_shapes_[r][1];
  }
}

template <>
void Transpose<3>::postprocess_cpu() {
  zisa::int_t k_offset = 0;
  for (int r = 0; r < size_; ++r) {
    for (zisa::int_t d = 0; d < to_.shape(0); ++d) {
      for (zisa::int_t i = 0; i < to_.shape(1); ++i) {
        for (zisa::int_t j = 0; j < to_.shape(2); ++j) {
          for (zisa::int_t k = 0; k < from_shapes_[r][1]; ++k) {
            to_(d, i, j, k + k_offset) = recvbuf_(r, d, i, j, k);
          }
        }
      }
    }
    k_offset += from_shapes_[r][1];
  }
}

template <int Dim>
void Transpose<Dim>::postprocess() {
  AZEBAN_PROFILE_START("Transpose::postprocess");
  if (location_ == zisa::device_type::cpu) {
    postprocess_cpu();
  }
#if ZISA_HAS_CUDA
  else if (location_ == zisa::device_type::cuda) {
    transpose_cuda_postprocess(
        recvbuf_, to_, from_shapes_.get(), to_shapes_.get(), rank_);
  }
#endif
  AZEBAN_PROFILE_STOP("Transpose::postprocess");
}

template class Transpose<2>;
template class Transpose<3>;

void transpose(const zisa::array_view<complex_t, 3> &dst,
               const zisa::array_const_view<complex_t, 3> &src,
               MPI_Comm comm) {
  Transpose<2> trans(comm, src, dst);
  const auto buffer_shape = trans.buffer_shape();
  auto sendbuf = zisa::array<complex_t, 4>(buffer_shape, trans.location());
  auto recvbuf = zisa::array<complex_t, 4>(buffer_shape, trans.location());
  trans.set_send_buffer(sendbuf);
  trans.set_recv_buffer(recvbuf);
  trans.eval();
}

void transpose(const zisa::array_view<complex_t, 4> &dst,
               const zisa::array_const_view<complex_t, 4> &src,
               MPI_Comm comm) {
  Transpose<3> trans(comm, src, dst);
  const auto buffer_shape = trans.buffer_shape();
  auto sendbuf = zisa::array<complex_t, 5>(buffer_shape, trans.location());
  auto recvbuf = zisa::array<complex_t, 5>(buffer_shape, trans.location());
  trans.set_send_buffer(sendbuf);
  trans.set_recv_buffer(recvbuf);
  trans.eval();
}

}

#endif
