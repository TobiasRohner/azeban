#include <azeban/copy_padded.hpp>

namespace azeban {

void copy_to_padded(const zisa::array_view<complex_t, 1> &dst,
                    const zisa::array_const_view<complex_t, 1> &src,
                    const complex_t &pad_value) {
  if (src.memory_location() == zisa::device_type::cpu
      && dst.memory_location() == zisa::device_type::cpu) {
    // TODO: Implement
    assert(false && "CPU to CPU padded copy not yet implemented");
  }
#ifdef ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda
           && dst.memory_location() == zisa::device_type::cuda) {
    copy_to_padded_cuda(dst, src, pad_value);
  }
#endif
  else {
    assert(false && "Unsupported combination of CPU and CUDA arrays");
  }
}

void copy_to_padded(const zisa::array_view<complex_t, 2> &dst,
                    const zisa::array_const_view<complex_t, 2> &src,
                    const complex_t &pad_value) {
  if (src.memory_location() == zisa::device_type::cpu
      && dst.memory_location() == zisa::device_type::cpu) {
    // TODO: Implement
    assert(false && "CPU to CPU padded copy not yet implemented");
  }
#ifdef ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda
           && dst.memory_location() == zisa::device_type::cuda) {
    copy_to_padded_cuda(dst, src, pad_value);
  }
#endif
  else {
    assert(false && "Unsupported combination of CPU and CUDA arrays");
  }
}

void copy_to_padded(const zisa::array_view<complex_t, 3> &dst,
                    const zisa::array_const_view<complex_t, 3> &src,
                    const complex_t &pad_value) {
  if (src.memory_location() == zisa::device_type::cpu
      && dst.memory_location() == zisa::device_type::cpu) {
    // TODO: Implement
    assert(false && "CPU to CPU padded copy not yet implemented");
  }
#ifdef ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda
           && dst.memory_location() == zisa::device_type::cuda) {
    copy_to_padded_cuda(dst, src, pad_value);
  }
#endif
  else {
    assert(false && "Unsupported combination of CPU and CUDA arrays");
  }
}

}
