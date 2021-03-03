
#ifndef BURGERS_H_
#define BURGERS_H_

#include "equation.hpp"
#include <azeban/config.hpp>
#include <azeban/fft.hpp>
#include <azeban/operations/convolve.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/equations/burgers_cuda.hpp>
#endif

namespace azeban {

template <typename SpectralViscosity>
class Burgers final : public Equation<complex_t, 1> {
  using super = Equation<complex_t, 1>;

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = 1;

  Burgers(zisa::int_t n,
          const SpectralViscosity &visc,
          zisa::device_type device = zisa::device_type::cpu)
      : device_(device), visc_(visc) {
    N_phys = n;
    N_fourier = N_phys / 2 + 1;
    N_phys_pad = 3. / 2 * N_phys + 1;
    N_fourier_pad = N_phys_pad / 2 + 1;
    u_hat_ = zisa::array<complex_t, 1>(zisa::shape_t<1>{N_fourier_pad}, device);
    u_ = zisa::array<real_t, 1>(zisa::shape_t<1>{N_phys_pad}, device);
    fft_ = make_fft(zisa::array_view<complex_t, 1>(u_hat_),
                    zisa::array_view<real_t, 1>(u_));
  }
  Burgers(const Burgers &) = delete;
  Burgers(Burgers &&) = default;
  virtual ~Burgers() override = default;
  Burgers &operator=(const Burgers &) = delete;
  Burgers &operator=(Burgers &&) = default;

  virtual void dudt(const zisa::array_view<scalar_t, dim_v> &u_hat) override {
    copy_to_padded(zisa::array_view<complex_t, 1>(u_hat_),
                   zisa::array_const_view<complex_t, 1>(u_hat),
                   complex_t(0));
    fft_->backward();
    real_t norm = N_phys_pad * N_phys;
    detail::scale_and_square(zisa::array_view<real_t, 1>(u_),
                             real_t(1.0 / std::sqrt(norm)));
    fft_->forward();
    if (device_ == zisa::device_type::cpu) {
      LOG_ERR("Not implemented yet");
    }
#if ZISA_HAS_CUDA
    else if (device_ == zisa::device_type::cuda) {
      burgers_cuda(u_hat, zisa::array_const_view<complex_t, 1>(u_hat_), visc_);
    }
#endif
    else {
      LOG_ERR("Unsupported memory_location");
    }
  }

private:
  zisa::int_t N_phys;
  zisa::int_t N_fourier;
  zisa::int_t N_phys_pad;
  zisa::int_t N_fourier_pad;
  zisa::device_type device_;
  zisa::array<complex_t, 1> u_hat_;
  zisa::array<real_t, 1> u_;
  std::shared_ptr<FFT<1>> fft_;
  SpectralViscosity visc_;
};

}

#endif
