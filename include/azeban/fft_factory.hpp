#ifndef FFT_FACTORY_H_
#define FFT_FACTORY_H_

#include <azeban/config.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/cufft.hpp>
#endif
#include <zisa/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<FFT<Dim>>
make_fft(const zisa::array_view<complex_t, Dim> &u_hat,
         const zisa::array_view<real_t, Dim> &u) {
#ifdef ZISA_HAS_CUDA
  if (u_hat.memory_location() == zisa::device_type::cuda
      && u.memory_location() == zisa::device_type::cuda) {
    return std::make_shared<CUFFT<Dim>>(u_hat, u);
  }
#endif
  assert(false && "Unsupported combination of memory loctions");
}

template <int Dim>
std::shared_ptr<FFT<Dim>>
make_fft(const zisa::array_view<complex_t, Dim + 1> &u_hat,
         const zisa::array_view<real_t, Dim + 1> &u) {
#ifdef ZISA_HAS_CUDA
  if (u_hat.memory_location() == zisa::device_type::cuda
      && u.memory_location() == zisa::device_type::cuda) {
    return std::make_shared<CUFFT<Dim>>(u_hat, u);
  }
#endif
  assert(false && "Unsupported combination of memory loctions");
}

}

#endif
