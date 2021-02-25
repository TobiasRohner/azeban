#ifndef COPY_PADDED_H_
#define COPY_PADDED_H_

#include <zisa/memory/array_view.hpp>
#ifdef ZISA_HAS_CUDA
#include "cuda/copy_padded_cuda.hpp"
#endif



namespace azeban {


template<typename T>
void copy_to_padded(const zisa::array_view<T, 1> &dst, const zisa::array_const_view<T, 1> &src, const T &pad_value) {
  if (src.memory_location() == zisa::device_type::cpu && dst.memory_location() == zisa::device_type::cpu) {
    // TODO: Implement
    assert(false && "CPU to CPU padded copy not yet implemented");
  }
#ifdef ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda && dst.memory_location() == zisa::device_type::cuda) {
    copy_to_padded_cuda(dst, src, pad_value);
  }
#endif
  else {
    assert(false && "Unsupported combination of CPU and CUDA arrays");
  }
}


template<typename T>
void copy_to_padded(const zisa::array_view<T, 2> &dst, const zisa::array_const_view<T, 2> &src, const T &pad_value) {
  if (src.memory_location() == zisa::device_type::cpu && dst.memory_location() == zisa::device_type::cpu) {
    // TODO: Implement
    assert(false && "CPU to CPU padded copy not yet implemented");
  }
#ifdef ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda && dst.memory_location() == zisa::device_type::cuda) {
    copy_to_padded_cuda(dst, src, pad_value);
  }
#endif
  else {
    assert(false && "Unsupported combination of CPU and CUDA arrays");
  }
}


template<typename T>
void copy_to_padded(const zisa::array_view<T, 3> &dst, const zisa::array_const_view<T, 3> &src, const T &pad_value) {
  if (src.memory_location() == zisa::device_type::cpu && dst.memory_location() == zisa::device_type::cpu) {
    // TODO: Implement
    assert(false && "CPU to CPU padded copy not yet implemented");
  }
#ifdef ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda && dst.memory_location() == zisa::device_type::cuda) {
    copy_to_padded_cuda(dst, src, pad_value);
  }
#endif
  else {
    assert(false && "Unsupported combination of CPU and CUDA arrays");
  }
}


template<typename T>
void copy_from_padded(const zisa::array_view<T, 1> &dst, const zisa::array_const_view<T, 1> &src) {
  if (src.memory_location() == zisa::device_type::cpu && dst.memory_location() == zisa::device_type::cpu) {
    // TODO: Implement
    assert(false && "CPU to CPU padded copy not yet implemented");
  }
#ifdef ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda && dst.memory_location() == zisa::device_type::cuda) {
    copy_from_padded_cuda(dst, src);
  }
#endif
  else {
    assert(false && "Unsupported combination of CPU and CUDA arrays");
  }
}


template<typename T>
void copy_from_padded(const zisa::array_view<T, 2> &dst, const zisa::array_const_view<T, 2> &src) {
  if (src.memory_location() == zisa::device_type::cpu && dst.memory_location() == zisa::device_type::cpu) {
    // TODO: Implement
    assert(false && "CPU to CPU padded copy not yet implemented");
  }
#ifdef ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda && dst.memory_location() == zisa::device_type::cuda) {
    copy_from_padded_cuda(dst, src);
  }
#endif
  else {
    assert(false && "Unsupported combination of CPU and CUDA arrays");
  }
}


template<typename T>
void copy_from_padded(const zisa::array_view<T, 3> &dst, const zisa::array_const_view<T, 3> &src) {
  if (src.memory_location() == zisa::device_type::cpu && dst.memory_location() == zisa::device_type::cpu) {
    // TODO: Implement
    assert(false && "CPU to CPU padded copy not yet implemented");
  }
#ifdef ZISA_HAS_CUDA
  else if (src.memory_location() == zisa::device_type::cuda && dst.memory_location() == zisa::device_type::cuda) {
    copy_from_padded_cuda(dst, src);
  }
#endif
  else {
    assert(false && "Unsupported combination of CPU and CUDA arrays");
  }
}


}



#endif
