#include <azeban/operations/fft_factory.hpp>
#include <azeban/operations/fftwfft.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/cufft.hpp>
#endif
#include <zisa/config.hpp>

namespace azeban {

template <int Dim, typename ScalarU>
static std::shared_ptr<FFT<Dim, ScalarU>>
make_fft_impl(const zisa::array_view<complex_t, Dim + 1> &u_hat,
              const zisa::array_view<ScalarU, Dim + 1> &u,
              int direction) {
  if (u_hat.memory_location() == zisa::device_type::cpu
      && u.memory_location() == zisa::device_type::cpu) {
    return std::make_shared<FFTWFFT<Dim, ScalarU>>(u_hat, u, direction);
  }
#ifdef ZISA_HAS_CUDA
  else if (u_hat.memory_location() == zisa::device_type::cuda
           && u.memory_location() == zisa::device_type::cuda) {
    return std::make_shared<CUFFT<Dim, ScalarU>>(u_hat, u, direction);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of memory loctions");
  }
}

template <int Dim>
std::shared_ptr<FFT<Dim, real_t>>
make_fft(const zisa::array_view<complex_t, Dim + 1> &u_hat,
         const zisa::array_view<real_t, Dim + 1> &u,
         int direction) {
  return make_fft_impl<Dim, real_t>(u_hat, u, direction);
}

template <int Dim>
std::shared_ptr<FFT<Dim, complex_t>>
make_fft(const zisa::array_view<complex_t, Dim + 1> &u_hat,
         const zisa::array_view<complex_t, Dim + 1> &u,
         int direction) {
  return make_fft_impl<Dim, complex_t>(u_hat, u, direction);
}

template std::shared_ptr<FFT<1, real_t>>
make_fft<1>(const zisa::array_view<complex_t, 2> &u_hat,
            const zisa::array_view<real_t, 2> &u,
            int direction);
template std::shared_ptr<FFT<2, real_t>>
make_fft<2>(const zisa::array_view<complex_t, 3> &u_hat,
            const zisa::array_view<real_t, 3> &u,
            int direction);
template std::shared_ptr<FFT<3, real_t>>
make_fft<3>(const zisa::array_view<complex_t, 4> &u_hat,
            const zisa::array_view<real_t, 4> &u,
            int direction);
template std::shared_ptr<FFT<1, complex_t>>
make_fft<1>(const zisa::array_view<complex_t, 2> &u_hat,
            const zisa::array_view<complex_t, 2> &u,
            int direction);
template std::shared_ptr<FFT<2, complex_t>>
make_fft<2>(const zisa::array_view<complex_t, 3> &u_hat,
            const zisa::array_view<complex_t, 3> &u,
            int direction);
template std::shared_ptr<FFT<3, complex_t>>
make_fft<3>(const zisa::array_view<complex_t, 4> &u_hat,
            const zisa::array_view<complex_t, 4> &u,
            int direction);

}
