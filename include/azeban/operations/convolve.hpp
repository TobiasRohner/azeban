#ifndef CONVOLVE_H_
#define CONVOLVE_H_

#include <azeban/copy_padded.hpp>
#include <azeban/fft.hpp>
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

template <int Dim>
zisa::array_view<complex_t, Dim>
component(const zisa::array_view<complex_t, Dim + 1> &arr, int dim) {
  zisa::shape_t<Dim> shape;
  for (int i = 0; i < Dim; ++i) {
    shape[i] = arr.shape(i + 1);
  }
  return {shape, arr.raw() + dim * zisa::product(shape), arr.memory_location()};
}

template <int Dim>
zisa::array_const_view<complex_t, Dim>
component(const zisa::array_const_view<complex_t, Dim + 1> &arr, int dim) {
  zisa::shape_t<Dim> shape;
  for (int i = 0; i < Dim; ++i) {
    shape[i] = arr.shape(i + 1);
  }
  return {shape, arr.raw() + dim * zisa::product(shape), arr.memory_location()};
}

}

template <int Dim>
void convolve_freq_domain(FFT<Dim> *fft,
                          const zisa::array_view<complex_t, Dim + 1> &u_hat) {
  const auto &u_hat_fft = fft->u_hat();

  for (int i = 0; i < Dim; ++i) {
    copy_to_padded(detail::component<Dim>(u_hat_fft, i),
                   zisa::array_const_view<complex_t, Dim>(
                       detail::component<Dim>(u_hat, i)),
                   complex_t(0));
  }

  fft->backward();
  real_t norm = zisa::product(fft->u().shape()) / fft->u().shape(0);
  detail::scale_and_square(fft->u(), real_t(1.0 / zisa::sqrt(norm)));
  fft->forward();

  for (int i = 0; i < Dim; ++i) {
    copy_from_padded(detail::component<Dim>(u_hat, i),
                     zisa::array_const_view<complex_t, Dim>(
                         detail::component<Dim>(u_hat_fft, i)));
  }
}

template <int Dim>
void convolve_freq_domain(FFT<Dim> *fft,
                          const zisa::array_view<complex_t, Dim> &u_hat) {
  zisa::shape_t<Dim + 1> shape;
  shape[0] = 1;
  for (zisa::int_t i = 0; i < Dim; ++i) {
    shape[i + 1] = u_hat.shape(i);
  }
  zisa::array_view<complex_t, Dim + 1> u_hat_new(
      shape, u_hat.raw(), u_hat.memory_location());
  convolve_freq_domain(fft, u_hat_new);
}

}

#endif
