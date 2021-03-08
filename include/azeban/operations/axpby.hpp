#ifndef AXPBY_H_
#define AXPBY_H_

#include <zisa/memory/array_view.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/axpby_cuda.hpp>
#endif

namespace azeban {

template <typename Scalar, int Dim>
void axpby(const Scalar &a,
           const zisa::array_const_view<Scalar, Dim> &x,
           const Scalar &b,
           const zisa::array_view<Scalar, Dim> &y) {
  assert(x.shape() == y.shape());

  const zisa::shape_t<1> flat_shape{zisa::product(x.shape())};
  const zisa::array_const_view<Scalar, 1> x_flat(
      flat_shape, x.raw(), x.memory_location());
  const zisa::array_view<Scalar, 1> y_flat(
      flat_shape, y.raw(), y.memory_location());

  if (x_flat.memory_location() == zisa::device_type::cpu
      && y_flat.memory_location() == zisa::device_type::cpu) {
    for (zisa::int_t i = 0; i < x_flat.shape(0); ++i) {
      y[i] = a * x[i] + b * y[i];
    }
  }
#if ZISA_HAS_CUDA
  else if (x_flat.memory_location() == zisa::device_type::cuda
           && y_flat.memory_location() == zisa::device_type::cuda) {
    axpby_cuda(a, x_flat, b, y_flat);
  }
#endif
  else {
    assert(false && "Unsupported combination of memory locations");
  }
}

}

#endif
