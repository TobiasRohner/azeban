#ifndef CONVOLVE_H_
#define CONVOLVE_H_

#include <azeban/copy_padded.hpp>
#include <azeban/fft.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/convolve_cuda.hpp>
#endif



namespace azeban {


namespace detail {

template<int Dim>
void square_and_scale(const zisa::array_view<real_t, Dim> &u, real_t scale) {
  const zisa::shape_t<1> flat_shape{zisa::product(u.shape())};
  const zisa::array_view<real_t, 1> flat(flat_shape, u.raw(), u.memory_location());
  if (flat.memory_location() == zisa::device_type::cpu) {
    for (zisa::int_t i = 0 ; i < flat.shape(0) ; ++i) {
      flat[i] = scale * u[i] * u[i];
    }
  }
#ifdef ZISA_HAS_CUDA
  else if (flat.memory_location() == zisa::device_type::cuda) {
    square_and_scale_cuda(flat, scale);
  }
#endif
  else {
    assert(false && "Unsupported memory location");
  }
}

}


template<typename Scalar, int Dim1, int... Dims, typename=std::enable_if_t<(... && (Dims == Dim1))>>
void convolve_freq_domain(FFT<Dim1> *fft,
			  const zisa::array_view<Scalar, Dim1> &u_hat1,
			  const zisa::array_view<Scalar, Dims>&... u_hats) {
  static_assert(Dim1 == 1, "convolve_freq_domain is only implemented for 1 dimension");
  assert(1+sizeof...(u_hats) == fft->shape(0));
  const auto& u_hat_fft = fft->u_hat();

  zisa::shape_t<Dim1> new_shape;
  for (zisa::int_t j = 0; j < Dim1 ; ++j) {
    new_shape[j] = u_hat_fft.shape(j+1);
  }

  const auto copy_to_fft = [&](const zisa::array_const_view<Scalar, Dim1>& arr, zisa::int_t i) {
    zisa::array_view<Scalar, Dim1> slice_fft(new_shape,
					     u_hat_fft.raw() + i*zisa::product(new_shape),
					     u_hat_fft.memory_location());
    copy_to_padded(slice_fft, arr, Scalar(0));
  };
  zisa::int_t i = 0;
  copy_to_fft(u_hat1, 0);
  (..., copy_to_fft(u_hats, ++i));

  fft->backward();
  // TODO: Compute this for arbitrary dimensions
  real_t norm = fft->u().shape(1);
  detail::square_and_scale(fft->u(), real_t(1.0/norm));
  fft->forward();

  const auto copy_from_fft = [&](const zisa::array_view<Scalar, Dim1>& arr, zisa::int_t i) {
    zisa::array_const_view<Scalar, Dim1> slice_fft(new_shape,
					           u_hat_fft.raw() + i*zisa::product(new_shape),
					           u_hat_fft.memory_location());
    copy_from_padded(arr, slice_fft);
  };
  i = 0;
  copy_from_fft(u_hat1, 0);
  (..., copy_from_fft(u_hats, ++i));
}


}



#endif
