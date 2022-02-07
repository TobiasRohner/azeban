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
#include <azeban/operations/copy_from_padded.hpp>
#include <azeban/operations/copy_from_padded_functions.hpp>
#include <azeban/profiler.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/copy_from_padded_cuda.hpp>
#endif

namespace azeban {

template <bool pad_x, int compact_dim>
void copy_from_padded_cpu(const zisa::array_view<complex_t, 1> &dst,
                          const zisa::array_const_view<complex_t, 1> &src) {
  for (zisa::int_t i = 0; i < dst.shape(0); ++i) {
    const auto src_shape = src.shape();
    const auto dst_shape = dst.shape();
    const unsigned long idx_dst
        = zisa::row_major<1>::linear_index(dst_shape, i);
    const unsigned long i_src
        = unpad_dim<pad_x, compact_dim == 0>(src_shape[0], dst_shape[0], i);
    const unsigned long idx_src
        = zisa::row_major<1>::linear_index(src_shape, i_src);
    dst[idx_dst] = src[idx_src];
  }
}

template <bool pad_x, bool pad_y, int compact_dim>
void copy_from_padded_cpu(const zisa::array_view<complex_t, 2> &dst,
                          const zisa::array_const_view<complex_t, 2> &src) {
  for (zisa::int_t i = 0; i < dst.shape(0); ++i) {
    for (zisa::int_t j = 0; j < dst.shape(1); ++j) {
      const auto src_shape = src.shape();
      const auto dst_shape = dst.shape();
      const unsigned long idx_dst
          = zisa::row_major<2>::linear_index(dst_shape, i, j);
      const unsigned long i_src
          = unpad_dim<pad_x, compact_dim == 0>(src_shape[0], dst_shape[0], i);
      const unsigned long j_src
          = unpad_dim<pad_y, compact_dim == 1>(src_shape[1], dst_shape[1], j);
      const unsigned long idx_src
          = zisa::row_major<2>::linear_index(src_shape, i_src, j_src);
      dst[idx_dst] = src[idx_src];
    }
  }
}

template <bool pad_x, bool pad_y, bool pad_z, int compact_dim>
void copy_from_padded_cpu(const zisa::array_view<complex_t, 3> &dst,
                          const zisa::array_const_view<complex_t, 3> &src) {
  for (zisa::int_t i = 0; i < dst.shape(0); ++i) {
    for (zisa::int_t j = 0; j < dst.shape(1); ++j) {
      for (zisa::int_t k = 0; k < dst.shape(2); ++k) {
        const auto src_shape = src.shape();
        const auto dst_shape = dst.shape();
        const unsigned long idx_dst
            = zisa::row_major<3>::linear_index(dst_shape, i, j, k);
        const unsigned long i_src
            = unpad_dim<pad_x, compact_dim == 0>(src_shape[0], dst_shape[0], i);
        const unsigned long j_src
            = unpad_dim<pad_y, compact_dim == 1>(src_shape[1], dst_shape[1], j);
        const unsigned long k_src
            = unpad_dim<pad_z, compact_dim == 2>(src_shape[2], dst_shape[2], k);
        const unsigned long idx_src
            = zisa::row_major<3>::linear_index(src_shape, i_src, j_src, k_src);
        dst[idx_dst] = src[idx_src];
      }
    }
  }
}

template <int compact_dim>
void copy_from_padded_cpu(bool pad_x,
                          const zisa::array_view<complex_t, 1> &dst,
                          const zisa::array_const_view<complex_t, 1> &src) {
  if (pad_x) {
    copy_from_padded_cpu<true, compact_dim>(dst, src);
  }
  if (!pad_x) {
    copy_from_padded_cpu<false, compact_dim>(dst, src);
  }
}

void copy_from_padded_cpu(bool pad_x,
                          int compact_dim,
                          const zisa::array_view<complex_t, 1> &dst,
                          const zisa::array_const_view<complex_t, 1> &src) {
  switch (compact_dim) {
  case 0:
    copy_from_padded_cpu<0>(pad_x, dst, src);
    return;
  default:
    copy_from_padded_cpu<-1>(pad_x, dst, src);
    return;
  }
}

template <int compact_dim>
void copy_from_padded_cpu(bool pad_x,
                          bool pad_y,
                          const zisa::array_view<complex_t, 2> &dst,
                          const zisa::array_const_view<complex_t, 2> &src) {
  if (pad_x && pad_y) {
    copy_from_padded_cpu<true, true, compact_dim>(dst, src);
  }
  if (pad_x && !pad_y) {
    copy_from_padded_cpu<true, false, compact_dim>(dst, src);
  }
  if (!pad_x && pad_y) {
    copy_from_padded_cpu<false, true, compact_dim>(dst, src);
  }
  if (!pad_x && !pad_y) {
    copy_from_padded_cpu<false, false, compact_dim>(dst, src);
  }
}

void copy_from_padded_cpu(bool pad_x,
                          bool pad_y,
                          int compact_dim,
                          const zisa::array_view<complex_t, 2> &dst,
                          const zisa::array_const_view<complex_t, 2> &src) {
  switch (compact_dim) {
  case 0:
    copy_from_padded_cpu<0>(pad_x, pad_y, dst, src);
    return;
  case 1:
    copy_from_padded_cpu<1>(pad_x, pad_y, dst, src);
    return;
  default:
    copy_from_padded_cpu<-1>(pad_x, pad_y, dst, src);
    return;
  }
}

template <int compact_dim>
void copy_from_padded_cpu(bool pad_x,
                          bool pad_y,
                          bool pad_z,
                          const zisa::array_view<complex_t, 3> &dst,
                          const zisa::array_const_view<complex_t, 3> &src) {
  if (pad_x && pad_y && pad_z) {
    copy_from_padded_cpu<true, true, true, compact_dim>(dst, src);
  }
  if (pad_x && pad_y && !pad_z) {
    copy_from_padded_cpu<true, true, false, compact_dim>(dst, src);
  }
  if (pad_x && !pad_y && pad_z) {
    copy_from_padded_cpu<true, false, true, compact_dim>(dst, src);
  }
  if (pad_x && !pad_y && !pad_z) {
    copy_from_padded_cpu<true, false, false, compact_dim>(dst, src);
  }
  if (!pad_x && pad_y && pad_z) {
    copy_from_padded_cpu<false, true, true, compact_dim>(dst, src);
  }
  if (!pad_x && pad_y && !pad_z) {
    copy_from_padded_cpu<false, true, false, compact_dim>(dst, src);
  }
  if (!pad_x && !pad_y && pad_z) {
    copy_from_padded_cpu<false, false, true, compact_dim>(dst, src);
  }
  if (!pad_x && !pad_y && !pad_z) {
    copy_from_padded_cpu<false, false, false, compact_dim>(dst, src);
  }
}

void copy_from_padded_cpu(bool pad_x,
                          bool pad_y,
                          bool pad_z,
                          int compact_dim,
                          const zisa::array_view<complex_t, 3> &dst,
                          const zisa::array_const_view<complex_t, 3> &src) {
  switch (compact_dim) {
  case 0:
    copy_from_padded_cpu<0>(pad_x, pad_y, pad_z, dst, src);
    return;
  case 1:
    copy_from_padded_cpu<1>(pad_x, pad_y, pad_z, dst, src);
    return;
  case 2:
    copy_from_padded_cpu<2>(pad_x, pad_y, pad_z, dst, src);
    return;
  default:
    copy_from_padded_cpu<-1>(pad_x, pad_y, pad_z, dst, src);
    return;
  }
}

void copy_from_padded(bool pad_x,
                      int compact_dim,
                      const zisa::array_view<complex_t, 1> &dst,
                      const zisa::array_const_view<complex_t, 1> &src) {
  AZEBAN_PROFILE_START("copy_from_padded");
  if (src.memory_location() == zisa::device_type::cpu
      && dst.memory_location() == zisa::device_type::cpu) {
    copy_from_padded_cpu(pad_x, compact_dim, dst, src);
  }
#if ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda
           && dst.memory_location() == zisa::device_type::cuda) {
    copy_from_padded_cuda(pad_x, compact_dim, dst, src);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of CPU and CUDA arrays");
  }
  AZEBAN_PROFILE_STOP("copy_from_padded");
}

void copy_from_padded(const zisa::array_view<complex_t, 1> &dst,
                      const zisa::array_const_view<complex_t, 1> &src) {
  copy_from_padded(true, 0, dst, src);
}

void copy_from_padded(bool pad_x,
                      bool pad_y,
                      int compact_dim,
                      const zisa::array_view<complex_t, 2> &dst,
                      const zisa::array_const_view<complex_t, 2> &src) {
  AZEBAN_PROFILE_START("copy_from_padded");
  if (src.memory_location() == zisa::device_type::cpu
      && dst.memory_location() == zisa::device_type::cpu) {
    copy_from_padded_cpu(pad_x, pad_y, compact_dim, dst, src);
  }
#if ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda
           && dst.memory_location() == zisa::device_type::cuda) {
    copy_from_padded_cuda(pad_x, pad_y, compact_dim, dst, src);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of CPU and CUDA arrays");
  }
  AZEBAN_PROFILE_STOP("copy_from_padded");
}

void copy_from_padded(const zisa::array_view<complex_t, 2> &dst,
                      const zisa::array_const_view<complex_t, 2> &src) {
  copy_from_padded(true, true, 1, dst, src);
}

void copy_from_padded(bool pad_x,
                      bool pad_y,
                      bool pad_z,
                      int compact_dim,
                      const zisa::array_view<complex_t, 3> &dst,
                      const zisa::array_const_view<complex_t, 3> &src) {
  AZEBAN_PROFILE_START("copy_from_padded");
  if (src.memory_location() == zisa::device_type::cpu
      && dst.memory_location() == zisa::device_type::cpu) {
    copy_from_padded_cpu(pad_x, pad_y, pad_z, compact_dim, dst, src);
  }
#if ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda
           && dst.memory_location() == zisa::device_type::cuda) {
    copy_from_padded_cuda(pad_x, pad_y, pad_z, compact_dim, dst, src);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of CPU and CUDA arrays");
  }
  AZEBAN_PROFILE_STOP("copy_from_padded");
}

void copy_from_padded(const zisa::array_view<complex_t, 3> &dst,
                      const zisa::array_const_view<complex_t, 3> &src) {
  copy_from_padded(true, true, true, 2, dst, src);
}

}
