#ifndef NORM_H_
#define NORM_H_

#include <zisa/config.hpp>
#include <zisa/memory/array_view.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/norm_cuda.hpp>
#endif

namespace azeban {

template <typename Scalar>
real_t norm(const zisa::array_const_view<Scalar, 1> &data, real_t p) {
  if (data.memory_location() == zisa::device_type::cpu) {
    real_t val = 0;
    for (zisa::int_t i = 0; i < data.shape(0); ++i) {
      using zisa::abs;
      val += zisa::pow(abs(data[i]), p);
    }
    return zisa::pow(val, real_t(1. / p));
  }
#if ZISA_HAS_CUDA
  else if (data.memory_location() == zisa::device_type::cuda) {
    return norm_cuda(data, p);
  }
#endif
  else {
    assert(false && "Unsupported memory location");
  }
}

}

#endif
