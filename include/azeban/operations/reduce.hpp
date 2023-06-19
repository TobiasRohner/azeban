#ifndef AZEBAN_OPERATIONS_REDUCE_HPP_
#define AZEBAN_OPERATIONS_REDUCE_HPP_

#include <zisa/config.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/reduce_cuda.hpp>
#endif

namespace azeban {

template <int Dim>
real_t reduce_sum(const zisa::array_const_view<real_t, Dim> &data) {
  if (data.memory_location() == zisa::device_type::cpu) {
    real_t val = 0;
    for (zisa::int_t i = 0; i < zisa::product(data.shape()); ++i) {
      val += data[i];
    }
    return val;
  }
#if ZISA_HAS_CUDA
  else if (data.memory_location() == zisa::device_type::cuda) {
    zisa::array_const_view<real_t, 1> view(
        zisa::shape_t<1>(zisa::product(data.shape())),
        data.raw(),
        data.memory_location());
    return reduce_sum_cuda(view);
  }
#endif
  else {
    LOG_ERR("Unsupported memory location");
  }
  // Make compiler happy
  return 0;
}

template <int Dim>
real_t reduce_sum(const zisa::array_view<real_t, Dim> &data) {
  return reduce_sum(zisa::array_const_view<real_t, Dim>(
      data.shape(), data.raw(), data.memory_location()));
}

template <int Dim>
real_t reduce_sum(const zisa::array<real_t, Dim> &data) {
  return reduce_sum(zisa::array_const_view<real_t, Dim>(
      data.shape(), data.raw(), data.device()));
}

}

#endif
