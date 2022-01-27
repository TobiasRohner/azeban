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
#include <azeban/operations/copy_padded.hpp>
#include <azeban/profiler.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/copy_padded_cuda.hpp>
#endif

namespace azeban {

template <bool pad_x>
void copy_to_padded_cpu(const zisa::array_view<complex_t, 1> &dst,
                        const zisa::array_const_view<complex_t, 1> &src,
                        const complex_t &pad_value) {
  for (zisa::int_t i = 0; i < src.shape(0); ++i) {
    dst[i] = src[i];
  }
  if (pad_x) {
    for (zisa::int_t i = src.shape(0); i < dst.shape(0); ++i) {
      dst[i] = pad_value;
    }
  }
}

template <bool pad_x, bool pad_y>
void copy_to_padded_cpu(const zisa::array_view<complex_t, 2> &dst,
                        const zisa::array_const_view<complex_t, 2> &src,
                        const complex_t &pad_value) {
  for (zisa::int_t i = 0; i < dst.shape(0); ++i) {
    for (zisa::int_t j = 0; j < dst.shape(1); ++j) {
      const auto src_shape = src.shape();
      const auto dst_shape = dst.shape();
      const int idx_dst = zisa::row_major<2>::linear_index(dst_shape, i, j);
      int i_src = i;
      if (pad_y) {
        if (j >= src_shape[1]) {
          dst[idx_dst] = pad_value;
          continue;
        }
      }
      if (pad_x) {
        if (i < src_shape[0] / 2 + 1) {
          i_src = i;
        } else if (i < src_shape[0] / 2 + 1 + dst_shape[0] - src_shape[0]) {
          dst[idx_dst] = pad_value;
          continue;
        } else {
          i_src = i + src_shape[0] - dst_shape[0];
        }
      }
      const int idx_src = zisa::row_major<2>::linear_index(src_shape, i_src, j);
      dst[idx_dst] = src[idx_src];
    }
  }
}

template <bool pad_x, bool pad_y, bool pad_z>
void copy_to_padded_cpu(const zisa::array_view<complex_t, 3> &dst,
                        const zisa::array_const_view<complex_t, 3> &src,
                        const complex_t &pad_value) {
  for (zisa::int_t i = 0; i < dst.shape(0); ++i) {
    for (zisa::int_t j = 0; j < dst.shape(1); ++j) {
      for (zisa::int_t k = 0; k < dst.shape(2); ++k) {
        const auto src_shape = src.shape();
        const auto dst_shape = dst.shape();
        const int idx_dst
            = zisa::row_major<3>::linear_index(dst_shape, i, j, k);
        int i_src = i;
        int j_src = j;
        if (pad_z) {
          if (k >= src_shape[2]) {
            dst[idx_dst] = pad_value;
            continue;
          }
        }
        if (pad_y) {
          if (j < src_shape[1] / 2 + 1) {
            j_src = j;
          } else if (j < src_shape[1] / 2 + 1 + dst_shape[1] - src_shape[1]) {
            dst[idx_dst] = pad_value;
            continue;
          } else {
            j_src = j + src_shape[1] - dst_shape[1];
          }
        }
        if (pad_x) {
          if (i < src_shape[0] / 2 + 1) {
            i_src = i;
          } else if (i < src_shape[0] / 2 + 1 + dst_shape[0] - src_shape[0]) {
            dst[idx_dst] = pad_value;
            continue;
          } else {
            i_src = i + src_shape[0] - dst_shape[0];
          }
        }
        const int idx_src
            = zisa::row_major<3>::linear_index(src_shape, i_src, j_src, k);
        dst[idx_dst] = src[idx_src];
      }
    }
  }
}

void copy_to_padded_cpu(const zisa::array_view<complex_t, 1> &dst,
                        const zisa::array_const_view<complex_t, 1> &src,
                        bool pad_x,
                        const complex_t &pad_value) {
  if (pad_x) {
    copy_to_padded_cpu<true>(dst, src, pad_value);
  }
  if (!pad_x) {
    copy_to_padded_cpu<false>(dst, src, pad_value);
  }
}

void copy_to_padded_cpu(const zisa::array_view<complex_t, 2> &dst,
                        const zisa::array_const_view<complex_t, 2> &src,
                        bool pad_x,
                        bool pad_y,
                        const complex_t &pad_value) {
  if (pad_x && pad_y) {
    copy_to_padded_cpu<true, true>(dst, src, pad_value);
  }
  if (pad_x && !pad_y) {
    copy_to_padded_cpu<true, false>(dst, src, pad_value);
  }
  if (!pad_x && pad_y) {
    copy_to_padded_cpu<false, true>(dst, src, pad_value);
  }
  if (!pad_x && !pad_y) {
    copy_to_padded_cpu<false, false>(dst, src, pad_value);
  }
}

void copy_to_padded_cpu(const zisa::array_view<complex_t, 3> &dst,
                        const zisa::array_const_view<complex_t, 3> &src,
                        bool pad_x,
                        bool pad_y,
                        bool pad_z,
                        const complex_t &pad_value) {
  if (pad_x && pad_y && pad_z) {
    copy_to_padded_cpu<true, true, true>(dst, src, pad_value);
  }
  if (pad_x && pad_y && !pad_z) {
    copy_to_padded_cpu<true, true, false>(dst, src, pad_value);
  }
  if (pad_x && !pad_y && pad_z) {
    copy_to_padded_cpu<true, false, true>(dst, src, pad_value);
  }
  if (pad_x && !pad_y && !pad_z) {
    copy_to_padded_cpu<true, false, false>(dst, src, pad_value);
  }
  if (!pad_x && pad_y && pad_z) {
    copy_to_padded_cpu<false, true, true>(dst, src, pad_value);
  }
  if (!pad_x && pad_y && !pad_z) {
    copy_to_padded_cpu<false, true, false>(dst, src, pad_value);
  }
  if (!pad_x && !pad_y && pad_z) {
    copy_to_padded_cpu<false, false, true>(dst, src, pad_value);
  }
  if (!pad_x && !pad_y && !pad_z) {
    copy_to_padded_cpu<false, false, false>(dst, src, pad_value);
  }
}

void copy_to_padded(const zisa::array_view<complex_t, 1> &dst,
                    const zisa::array_const_view<complex_t, 1> &src,
                    bool pad_x,
                    const complex_t &pad_value) {
  AZEBAN_PROFILE_START("copy_to_padded");
  if (src.memory_location() == zisa::device_type::cpu
      && dst.memory_location() == zisa::device_type::cpu) {
    copy_to_padded_cpu(dst, src, pad_x, pad_value);
  }
#if ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda
           && dst.memory_location() == zisa::device_type::cuda) {
    copy_to_padded_cuda(dst, src, pad_x, pad_value);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of CPU and CUDA arrays");
  }
  AZEBAN_PROFILE_STOP("copy_to_padded");
}

void copy_to_padded(const zisa::array_view<complex_t, 1> &dst,
                    const zisa::array_const_view<complex_t, 1> &src,
                    const complex_t &pad_value) {
  copy_to_padded(dst, src, true, pad_value);
}

void copy_to_padded(const zisa::array_view<complex_t, 2> &dst,
                    const zisa::array_const_view<complex_t, 2> &src,
                    bool pad_x,
                    bool pad_y,
                    const complex_t &pad_value) {
  AZEBAN_PROFILE_START("copy_to_padded");
  if (src.memory_location() == zisa::device_type::cpu
      && dst.memory_location() == zisa::device_type::cpu) {
    copy_to_padded_cpu(dst, src, pad_x, pad_y, pad_value);
  }
#if ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda
           && dst.memory_location() == zisa::device_type::cuda) {
    copy_to_padded_cuda(dst, src, pad_x, pad_y, pad_value);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of CPU and CUDA arrays");
  }
  AZEBAN_PROFILE_STOP("copy_to_padded");
}

void copy_to_padded(const zisa::array_view<complex_t, 2> &dst,
                    const zisa::array_const_view<complex_t, 2> &src,
                    const complex_t &pad_value) {
  copy_to_padded(dst, src, true, true, pad_value);
}

void copy_to_padded(const zisa::array_view<complex_t, 3> &dst,
                    const zisa::array_const_view<complex_t, 3> &src,
                    bool pad_x,
                    bool pad_y,
                    bool pad_z,
                    const complex_t &pad_value) {
  AZEBAN_PROFILE_START("copy_to_padded");
  if (src.memory_location() == zisa::device_type::cpu
      && dst.memory_location() == zisa::device_type::cpu) {
    copy_to_padded_cpu(dst, src, pad_x, pad_y, pad_z, pad_value);
  }
#if ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda
           && dst.memory_location() == zisa::device_type::cuda) {
    copy_to_padded_cuda(dst, src, pad_x, pad_y, pad_z, pad_value);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of CPU and CUDA arrays");
  }
  AZEBAN_PROFILE_STOP("copy_to_padded");
}

void copy_to_padded(const zisa::array_view<complex_t, 3> &dst,
                    const zisa::array_const_view<complex_t, 3> &src,
                    const complex_t &pad_value) {
  copy_to_padded(dst, src, true, true, true, pad_value);
}

}
