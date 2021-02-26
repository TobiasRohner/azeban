#ifndef AXPY_H_
#define AXPY_H_

#include <zisa/memory/array_view.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/axpy_cuda.hpp>
#endif



namespace azeban {


template<typename Scalar, int Dim>
void axpy(const Scalar &a, const zisa::array_const_view<Scalar, Dim> &x, const zisa::array_view<Scalar, Dim> &y) {
  assert(x.shape() == y.shape());

  const zisa::shape_t<1> flat_shape{zisa::product(x.shape())};
  const zisa::array_const_view<Scalar, Dim> x_flat(flat_shape, x.raw(), x.memory_location());
  const zisa::array_view<Scalar, Dim> y_flat(flat_shape, y.raw(), y.memory_location());

  if (x_flat.memory_location() == zisa::device_type::cpu && y_flat.memory_location() == zisa::device_type::cpu) {
    for (zisa::int_t i = 0 ; i < x_flat.shape(0) ; ++i) {
      y[i] += a * x[i];
    }
  }
#ifdef ZISA_HAS_CUDA
  else if (x_flat.memory_location() == zisa::device_type::cuda && y_flat.memory_location() == zisa::device_type::cuda) {
    axpy_cuda(a, x_flat, y_flat);
  }
#endif
  else {
    assert(false && "Unsupported combination of memory locations");
  }
}


}



#endif
