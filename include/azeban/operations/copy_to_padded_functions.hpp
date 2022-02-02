#ifndef AZEBAN_OPERATIONS_COPY_TO_PADDED_FUNCTIONS_HPP_
#define AZEBAN_OPERATIONS_COPY_TO_PADDED_FUNCTIONS_HPP_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
ANY_DEVICE_INLINE bool
pad_compact_dim(const zisa::array_view<complex_t, Dim> &dst,
                int idx_dst,
                int src_shape,
                int i,
                const complex_t &pad_value,
                int *i_src) {
  if (i < src_shape) {
    *i_src = i;
    return false;
  } else {
    dst[idx_dst] = pad_value;
    return true;
  }
}

template <int Dim>
ANY_DEVICE_INLINE bool pad_full_dim(const zisa::array_view<complex_t, Dim> &dst,
                                    int idx_dst,
                                    int src_shape,
                                    int dst_shape,
                                    int i,
                                    const complex_t &pad_value,
                                    int *i_src) {
  if (i < src_shape / 2 + 1) {
    *i_src = i;
    return false;
  } else if (i < src_shape / 2 + 1 + dst_shape - src_shape) {
    dst[idx_dst] = pad_value;
    return true;
  } else {
    *i_src = i + src_shape - dst_shape;
    return false;
  }
}

template <bool pad, bool compact, int Dim>
ANY_DEVICE_INLINE bool pad_dim(const zisa::array_view<complex_t, Dim> &dst,
                               int idx_dst,
                               int src_shape,
                               int dst_shape,
                               int i,
                               const complex_t &pad_value,
                               int *i_src) {
  if (pad) {
    if (compact) {
      return pad_compact_dim(dst, idx_dst, src_shape, i, pad_value, i_src);
    } else {
      return pad_full_dim(
          dst, idx_dst, src_shape, dst_shape, i, pad_value, i_src);
    }
  }
  return false;
}

}

#endif
