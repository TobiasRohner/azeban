#include <azeban/operations/fft_factory.hpp>
#include <azeban/operations/fftwfft.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/cufft.hpp>
#endif
#include <zisa/config.hpp>

namespace azeban {

template <int Dim, typename ScalarU, typename... T>
static std::shared_ptr<FFT<Dim, ScalarU>>
make_fft_impl(zisa::device_type device, int direction, T... transform) {
  static_assert(sizeof...(T) == Dim || sizeof...(T) == 0, "");
  static_assert((... && std::is_convertible_v<T, bool>), "");
  if (device == zisa::device_type::cpu) {
    return std::make_shared<FFTWFFT<Dim, ScalarU>>(direction, transform...);
  }
#ifdef ZISA_HAS_CUDA
  else if (device == zisa::device_type::cuda) {
    return std::make_shared<CUFFT<Dim, ScalarU>>(direction, transform...);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of memory loctions");
  }
}

template <int Dim, typename ScalarU>
static std::shared_ptr<FFT<Dim, ScalarU>>
make_fft_impl(const zisa::array_view<complex_t, Dim + 1> &u_hat,
              const zisa::array_view<ScalarU, Dim + 1> &u,
              int direction) {
  LOG_ERR_IF(u_hat.memory_location() != u.memory_location(),
             "Memory locations mismatch");
  const zisa::device_type device = u.memory_location();
  std::shared_ptr<FFT<Dim, ScalarU>> fft
      = make_fft_impl<Dim, ScalarU>(device, direction);
  fft->initialize(u_hat, u);
  return fft;
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

template <int Dim, typename ScalarU, typename>
std::shared_ptr<FFT<Dim, ScalarU>>
make_fft(zisa::device_type device, int direction, bool transform_x) {
  return make_fft_impl<Dim, ScalarU>(device, direction, transform_x);
}

template <int Dim, typename ScalarU, typename>
std::shared_ptr<FFT<Dim, ScalarU>> make_fft(zisa::device_type device,
                                            int direction,
                                            bool transform_x,
                                            bool transform_y) {
  return make_fft_impl<Dim, ScalarU>(
      device, direction, transform_x, transform_y);
}

template <int Dim, typename ScalarU, typename>
std::shared_ptr<FFT<Dim, ScalarU>> make_fft(zisa::device_type device,
                                            int direction,
                                            bool transform_x,
                                            bool transform_y,
                                            bool transform_z) {
  return make_fft_impl<Dim, ScalarU>(
      device, direction, transform_x, transform_y, transform_z);
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
template std::shared_ptr<FFT<1, real_t>> make_fft<1, real_t, void>(
    zisa::device_type device, int direction, bool transform_x);
template std::shared_ptr<FFT<2, real_t>>
make_fft<2, real_t, void>(zisa::device_type device,
                          int direction,
                          bool transform_x,
                          bool transform_y);
template std::shared_ptr<FFT<3, real_t>>
make_fft<3, real_t, void>(zisa::device_type device,
                          int direction,
                          bool transform_x,
                          bool transform_y,
                          bool transform_z);
template std::shared_ptr<FFT<1, complex_t>> make_fft<1, complex_t, void>(
    zisa::device_type device, int direction, bool transform_x);
template std::shared_ptr<FFT<2, complex_t>>
make_fft<2, complex_t, void>(zisa::device_type device,
                             int direction,
                             bool transform_x,
                             bool transform_y);
template std::shared_ptr<FFT<3, complex_t>>
make_fft<3, complex_t, void>(zisa::device_type device,
                             int direction,
                             bool transform_x,
                             bool transform_y,
                             bool transform_z);

}
