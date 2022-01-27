#ifndef AZEBAN_OPERATIONS_COPY_FROM_PADDED_FUNCTIONS_HPP_
#define AZEBAN_OPERATIONS_COPY_FROM_PADDED_FUNCTIONS_HPP_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

ANY_DEVICE_INLINE int unpad_compact_dim(int i) { return i; }

ANY_DEVICE_INLINE int unpad_full_dim(int src_shape, int dst_shape, int i) {
  if (i < dst_shape / 2 + 1) {
    return i;
  } else {
    return i + src_shape - dst_shape;
  }
}

template <bool pad, bool compact>
ANY_DEVICE_INLINE int unpad_dim(int src_shape, int dst_shape, int i) {
  if (pad) {
    if (compact) {
      return unpad_compact_dim(i);
    } else {
      return unpad_full_dim(src_shape, dst_shape, i);
    }
  }
  return i;
}

}

#endif
