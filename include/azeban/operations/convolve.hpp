#ifndef CONVOLVE_H_
#define CONVOLVE_H_

#include <azeban/operations/copy_padded.hpp>
#include <azeban/operations/fft.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/convolve_cuda.hpp>
#endif

namespace azeban {

namespace detail {

template <int Dim>
void scale_and_square(const zisa::array_view<real_t, Dim> &u, real_t scale) {
  const zisa::shape_t<1> flat_shape{zisa::product(u.shape())};
  const zisa::array_view<real_t, 1> flat(
      flat_shape, u.raw(), u.memory_location());
  if (flat.memory_location() == zisa::device_type::cpu) {
    for (zisa::int_t i = 0; i < flat.shape(0); ++i) {
      const real_t ui_scaled = scale * u[i];
      flat[i] = ui_scaled * ui_scaled;
    }
  }
#ifdef ZISA_HAS_CUDA
  else if (flat.memory_location() == zisa::device_type::cuda) {
    scale_and_square_cuda(flat, scale);
  }
#endif
  else {
    assert(false && "Unsupported memory location");
  }
}

}

}

#endif
