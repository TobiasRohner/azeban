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
#include <azeban/operations/copy_to_padded.hpp>
#include <azeban/operations/copy_to_padded_functions.hpp>
#include <azeban/profiler.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/copy_to_padded_cuda.hpp>
#endif

namespace azeban {

template <bool pad_x, int compact_dim>
void copy_to_padded_cpu(const zisa::array_view<complex_t, 1> &dst,
                        const zisa::array_const_view<complex_t, 1> &src,
                        const complex_t &pad_value) {
  for (zisa::int_t i = 0; i < dst.shape(0); ++i) {
    const auto src_shape = src.shape();
    const auto dst_shape = dst.shape();
    const unsigned long idx_dst
        = zisa::row_major<1>::linear_index(dst_shape, i);
    unsigned long i_src = i;
    if (pad_dim<pad_x, compact_dim == 0>(
            dst, idx_dst, src_shape[0], dst_shape[0], i, pad_value, &i_src)) {
      continue;
    }
    const unsigned long idx_src
        = zisa::row_major<1>::linear_index(src_shape, i_src);
    dst[idx_dst] = src[idx_src];
  }
}

template <bool pad_x, bool pad_y, int compact_dim>
void copy_to_padded_cpu(const zisa::array_view<complex_t, 2> &dst,
                        const zisa::array_const_view<complex_t, 2> &src,
                        const complex_t &pad_value) {
  for (zisa::int_t i = 0; i < dst.shape(0); ++i) {
    for (zisa::int_t j = 0; j < dst.shape(1); ++j) {
      const auto src_shape = src.shape();
      const auto dst_shape = dst.shape();
      const unsigned long idx_dst
          = zisa::row_major<2>::linear_index(dst_shape, i, j);
      unsigned long i_src = i;
      unsigned long j_src = j;
      if (pad_dim<pad_x, compact_dim == 0>(
              dst, idx_dst, src_shape[0], dst_shape[0], i, pad_value, &i_src)) {
        continue;
      }
      if (pad_dim<pad_y, compact_dim == 1>(
              dst, idx_dst, src_shape[1], dst_shape[1], j, pad_value, &j_src)) {
        continue;
      }
      const unsigned long idx_src
          = zisa::row_major<2>::linear_index(src_shape, i_src, j_src);
      dst[idx_dst] = src[idx_src];
    }
  }
}

template <bool pad_x, bool pad_y, bool pad_z, int compact_dim>
void copy_to_padded_cpu(const zisa::array_view<complex_t, 3> &dst,
                        const zisa::array_const_view<complex_t, 3> &src,
                        const complex_t &pad_value) {
  for (zisa::int_t i = 0; i < dst.shape(0); ++i) {
    for (zisa::int_t j = 0; j < dst.shape(1); ++j) {
      for (zisa::int_t k = 0; k < dst.shape(2); ++k) {
        const auto src_shape = src.shape();
        const auto dst_shape = dst.shape();
        const unsigned long idx_dst
            = zisa::row_major<3>::linear_index(dst_shape, i, j, k);
        unsigned long i_src = i;
        unsigned long j_src = j;
        unsigned long k_src = k;
        if (pad_dim<pad_x, compact_dim == 0>(dst,
                                             idx_dst,
                                             src_shape[0],
                                             dst_shape[0],
                                             i,
                                             pad_value,
                                             &i_src)) {
          continue;
        }
        if (pad_dim<pad_y, compact_dim == 1>(dst,
                                             idx_dst,
                                             src_shape[1],
                                             dst_shape[1],
                                             j,
                                             pad_value,
                                             &j_src)) {
          continue;
        }
        if (pad_dim<pad_z, compact_dim == 2>(dst,
                                             idx_dst,
                                             src_shape[2],
                                             dst_shape[2],
                                             k,
                                             pad_value,
                                             &k_src)) {
          continue;
        }
        const unsigned long idx_src
            = zisa::row_major<3>::linear_index(src_shape, i_src, j_src, k_src);
        dst[idx_dst] = src[idx_src];
      }
    }
  }
}

template <int compact_dim>
void copy_to_padded_cpu(bool pad_x,
                        const zisa::array_view<complex_t, 1> &dst,
                        const zisa::array_const_view<complex_t, 1> &src,
                        const complex_t &pad_value) {
  if (pad_x) {
    copy_to_padded_cpu<true, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x) {
    copy_to_padded_cpu<false, compact_dim>(dst, src, pad_value);
  }
}

void copy_to_padded_cpu(bool pad_x,
                        int compact_dim,
                        const zisa::array_view<complex_t, 1> &dst,
                        const zisa::array_const_view<complex_t, 1> &src,
                        const complex_t &pad_value) {
  switch (compact_dim) {
  case 0:
    copy_to_padded_cpu<0>(pad_x, dst, src, pad_value);
    return;
  default:
    copy_to_padded_cpu<-1>(pad_x, dst, src, pad_value);
    return;
  }
}

template <int compact_dim>
void copy_to_padded_cpu(bool pad_x,
                        bool pad_y,
                        const zisa::array_view<complex_t, 2> &dst,
                        const zisa::array_const_view<complex_t, 2> &src,
                        const complex_t &pad_value) {
  if (pad_x && pad_y) {
    copy_to_padded_cpu<true, true, compact_dim>(dst, src, pad_value);
  }
  if (pad_x && !pad_y) {
    copy_to_padded_cpu<true, false, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x && pad_y) {
    copy_to_padded_cpu<false, true, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x && !pad_y) {
    copy_to_padded_cpu<false, false, compact_dim>(dst, src, pad_value);
  }
}

void copy_to_padded_cpu(bool pad_x,
                        bool pad_y,
                        int compact_dim,
                        const zisa::array_view<complex_t, 2> &dst,
                        const zisa::array_const_view<complex_t, 2> &src,
                        const complex_t &pad_value) {
  switch (compact_dim) {
  case 0:
    copy_to_padded_cpu<0>(pad_x, pad_y, dst, src, pad_value);
    return;
  case 1:
    copy_to_padded_cpu<1>(pad_x, pad_y, dst, src, pad_value);
    return;
  default:
    copy_to_padded_cpu<-1>(pad_x, pad_y, dst, src, pad_value);
    return;
  }
}

template <int compact_dim>
void copy_to_padded_cpu(bool pad_x,
                        bool pad_y,
                        bool pad_z,
                        const zisa::array_view<complex_t, 3> &dst,
                        const zisa::array_const_view<complex_t, 3> &src,
                        const complex_t &pad_value) {
  if (pad_x && pad_y && pad_z) {
    copy_to_padded_cpu<true, true, true, compact_dim>(dst, src, pad_value);
  }
  if (pad_x && pad_y && !pad_z) {
    copy_to_padded_cpu<true, true, false, compact_dim>(dst, src, pad_value);
  }
  if (pad_x && !pad_y && pad_z) {
    copy_to_padded_cpu<true, false, true, compact_dim>(dst, src, pad_value);
  }
  if (pad_x && !pad_y && !pad_z) {
    copy_to_padded_cpu<true, false, false, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x && pad_y && pad_z) {
    copy_to_padded_cpu<false, true, true, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x && pad_y && !pad_z) {
    copy_to_padded_cpu<false, true, false, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x && !pad_y && pad_z) {
    copy_to_padded_cpu<false, false, true, compact_dim>(dst, src, pad_value);
  }
  if (!pad_x && !pad_y && !pad_z) {
    copy_to_padded_cpu<false, false, false, compact_dim>(dst, src, pad_value);
  }
}

void copy_to_padded_cpu(bool pad_x,
                        bool pad_y,
                        bool pad_z,
                        int compact_dim,
                        const zisa::array_view<complex_t, 3> &dst,
                        const zisa::array_const_view<complex_t, 3> &src,
                        const complex_t &pad_value) {
  switch (compact_dim) {
  case 0:
    copy_to_padded_cpu<0>(pad_x, pad_y, pad_z, dst, src, pad_value);
    return;
  case 1:
    copy_to_padded_cpu<1>(pad_x, pad_y, pad_z, dst, src, pad_value);
    return;
  case 2:
    copy_to_padded_cpu<2>(pad_x, pad_y, pad_z, dst, src, pad_value);
    return;
  default:
    copy_to_padded_cpu<-1>(pad_x, pad_y, pad_z, dst, src, pad_value);
    return;
  }
}

void copy_to_padded(bool pad_x,
                    int compact_dim,
                    const zisa::array_view<complex_t, 1> &dst,
                    const zisa::array_const_view<complex_t, 1> &src,
                    const complex_t &pad_value) {
  ProfileHost profile("copy_to_padded");
  if (src.memory_location() == zisa::device_type::cpu
      && dst.memory_location() == zisa::device_type::cpu) {
    copy_to_padded_cpu(pad_x, compact_dim, dst, src, pad_value);
  }
#if ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda
           && dst.memory_location() == zisa::device_type::cuda) {
    copy_to_padded_cuda(pad_x, compact_dim, dst, src, pad_value);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of CPU and CUDA arrays");
  }
}

void copy_to_padded(const zisa::array_view<complex_t, 1> &dst,
                    const zisa::array_const_view<complex_t, 1> &src,
                    const complex_t &pad_value) {
  copy_to_padded(true, 0, dst, src, pad_value);
}

void copy_to_padded(bool pad_x,
                    bool pad_y,
                    int compact_dim,
                    const zisa::array_view<complex_t, 2> &dst,
                    const zisa::array_const_view<complex_t, 2> &src,
                    const complex_t &pad_value) {
  ProfileHost profile("copy_to_padded");
  if (src.memory_location() == zisa::device_type::cpu
      && dst.memory_location() == zisa::device_type::cpu) {
    copy_to_padded_cpu(pad_x, pad_y, compact_dim, dst, src, pad_value);
  }
#if ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda
           && dst.memory_location() == zisa::device_type::cuda) {
    copy_to_padded_cuda(pad_x, pad_y, compact_dim, dst, src, pad_value);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of CPU and CUDA arrays");
  }
}

void copy_to_padded(const zisa::array_view<complex_t, 2> &dst,
                    const zisa::array_const_view<complex_t, 2> &src,
                    const complex_t &pad_value) {
  copy_to_padded(true, true, 1, dst, src, pad_value);
}

void copy_to_padded(bool pad_x,
                    bool pad_y,
                    bool pad_z,
                    int compact_dim,
                    const zisa::array_view<complex_t, 3> &dst,
                    const zisa::array_const_view<complex_t, 3> &src,
                    const complex_t &pad_value) {
  ProfileHost profile("copy_to_padded");
  if (src.memory_location() == zisa::device_type::cpu
      && dst.memory_location() == zisa::device_type::cpu) {
    copy_to_padded_cpu(pad_x, pad_y, pad_z, compact_dim, dst, src, pad_value);
  }
#if ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda
           && dst.memory_location() == zisa::device_type::cuda) {
    copy_to_padded_cuda(pad_x, pad_y, pad_z, compact_dim, dst, src, pad_value);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of CPU and CUDA arrays");
  }
}

void copy_to_padded(const zisa::array_view<complex_t, 3> &dst,
                    const zisa::array_const_view<complex_t, 3> &src,
                    const complex_t &pad_value) {
  copy_to_padded(true, true, true, 2, dst, src, pad_value);
}

}
