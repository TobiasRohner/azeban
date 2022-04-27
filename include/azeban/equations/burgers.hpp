/*
 * This file is part of azeban (https://github.com/TobiasRohner/azeban).
 * Copyright (c) 2021 Tobias Rohner.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BURGERS_H_
#define BURGERS_H_

#include "equation.hpp"
#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <azeban/operations/convolve.hpp>
#include <azeban/operations/fft_factory.hpp>
#include <azeban/operations/norm.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/equations/burgers_cuda.hpp>
#endif

namespace azeban {

template <typename SpectralViscosity>
class Burgers final : public Equation<1> {
  using super = Equation<1>;

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = 1;

  Burgers(const Grid<1> &grid,
          const SpectralViscosity &visc,
          zisa::device_type device)
      : super(grid), device_(device), u_max_(0), visc_(visc) {
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
  dudt(const zisa::array_view<scalar_t, dim_v + 1> &dudt_hat,
       const zisa::array_const_view<scalar_t, dim_v + 1> &u_hat) override {
    copy_to_padded(
        zisa::array_view<complex_t, 1>(
            zisa::shape_t<1>(u_hat_.shape(1)), u_hat_.raw(), u_hat_.device()),
        zisa::array_const_view<complex_t, 1>(zisa::shape_t<1>(u_hat.shape(1)),
                                             u_hat.raw(),
                                             u_hat.memory_location()),
        complex_t(0));
    fft_->backward();
    u_max_ = max_norm(u_);
    real_t norm = grid_.N_phys_pad * grid_.N_phys;
    detail::scale_and_square(zisa::array_view<real_t, 2>(u_),
                             real_t(1.0 / std::sqrt(norm)));
    fft_->forward();
    if (device_ == zisa::device_type::cpu) {
      for (zisa::int_t k = 0; k < u_hat.shape(1); ++k) {
        const real_t k_ = 2 * zisa::pi * k;
        const real_t v = visc_.eval(k_);
        dudt_hat[k] = complex_t(0, -k_ / 2) * u_hat_[k] + v * u_hat[k];
      }
    }
#if ZISA_HAS_CUDA
    else if (device_ == zisa::device_type::cuda) {
      burgers_cuda(
          dudt_hat, u_hat, zisa::array_const_view<complex_t, 2>(u_hat_), visc_);
    }
#endif
    else {
      LOG_ERR("Unsupported memory_location");
    }
  }

  virtual real_t dt() const override { return 1. / u_max_; }

  virtual int n_vars() const override { return 1; }

private:
  zisa::device_type device_;
  real_t u_max_;
  zisa::array<complex_t, 2> u_hat_;
  zisa::array<real_t, 2> u_;
  std::shared_ptr<FFT<1>> fft_;
  SpectralViscosity visc_;
};

}

#endif
