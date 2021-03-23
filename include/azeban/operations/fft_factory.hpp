#ifndef FFT_FACTORY_H_
#define FFT_FACTORY_H_

#include <azeban/config.hpp>
#include <azeban/operations/fftwfft.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/cufft.hpp>
#endif
#include <zisa/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<FFT<Dim>>
make_fft(const zisa::array_view<complex_t, Dim + 1> &u_hat,
         const zisa::array_view<real_t, Dim + 1> &u,
         int direction = FFT_FORWARD | FFT_BACKWARD) {
  if (u_hat.memory_location() == zisa::device_type::cpu
      && u.memory_location() == zisa::device_type::cpu) {
    return std::make_shared<FFTWFFT<Dim>>(u_hat, u, direction);
  }
#ifdef ZISA_HAS_CUDA
  else if (u_hat.memory_location() == zisa::device_type::cuda
           && u.memory_location() == zisa::device_type::cuda) {
    return std::make_shared<CUFFT<Dim>>(u_hat, u, direction);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of memory loctions");
  }
}

}

#endif
