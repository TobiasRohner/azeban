
#ifndef BURGERS_H_
#define BURGERS_H_

#include "equation.hpp"
#include <azeban/config.hpp>
#include <azeban/operations/convolve.hpp>
#include <azeban/operations/fft.hpp>
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

  Burgers(const Grid<1> &grid,
          const SpectralViscosity &visc,
          zisa::device_type device)
      : super(grid), device_(device), visc_(visc) {
    u_hat_ = grid.make_array_fourier_pad(1, device);
    u_ = grid.make_array_phys_pad(1, device);
    fft_ = make_fft<1>(u_hat_, u_);
  }
  Burgers(const Burgers &) = delete;
  Burgers(Burgers &&) = default;
  virtual ~Burgers() override = default;
  Burgers &operator=(const Burgers &) = delete;
  Burgers &operator=(Burgers &&) = default;

  virtual void
  dudt(const zisa::array_view<scalar_t, dim_v + 1> &u_hat) override {
    copy_to_padded(
        zisa::array_view<complex_t, 1>(
            zisa::shape_t<1>(u_hat_.shape(1)), u_hat_.raw(), u_hat_.device()),
        zisa::array_const_view<complex_t, 1>(zisa::shape_t<1>(u_hat.shape(1)),
                                             u_hat.raw(),
                                             u_hat.memory_location()),
        complex_t(0));
    fft_->backward();
    real_t norm = grid_.N_phys_pad * grid_.N_phys;
    detail::scale_and_square(zisa::array_view<real_t, 2>(u_),
                             real_t(1.0 / std::sqrt(norm)));
    fft_->forward();
    if (device_ == zisa::device_type::cpu) {
      for (zisa::int_t k = 0; k < u_hat.shape(1); ++k) {
        const real_t k_ = 2 * zisa::pi * k;
        const real_t v = visc_.eval(k_);
        u_hat[k] = complex_t(0, -k_ / 2) * u_hat_[k] + v * u_hat[k];
      }
    }
#if ZISA_HAS_CUDA
    else if (device_ == zisa::device_type::cuda) {
      burgers_cuda(u_hat, zisa::array_const_view<complex_t, 2>(u_hat_), visc_);
    }
#endif
    else {
      LOG_ERR("Unsupported memory_location");
    }
  }

  using super::grid;
  virtual int n_vars() const override { return 1; }

private:
  zisa::device_type device_;
  zisa::array<complex_t, 2> u_hat_;
  zisa::array<real_t, 2> u_;
  std::shared_ptr<FFT<1>> fft_;
  SpectralViscosity visc_;
};

}

#endif
