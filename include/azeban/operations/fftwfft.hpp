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
#ifndef FFTWFFT_H_
#define FFTWFFT_H_

#include <azeban/operations/fft.hpp>
#include <azeban/profiler.hpp>
#include <fftw3.h>

namespace azeban {

template <int Dim>
class FFTWFFT final : public FFT<Dim> {
  using super = FFT<Dim>;

  static constexpr bool is_double = std::is_same_v<real_t, double>;
  using fftw_complex_t
      = std::conditional_t<is_double, fftw_complex, fftwf_complex>;
  using fftw_plan_t = std::conditional_t<is_double, fftw_plan, fftwf_plan>;

  static_assert(
      std::is_same_v<real_t,
                     std::decay_t<decltype(std::declval<fftw_complex_t>()[0])>>,
      "Real type has the wrong precision for FFTW");
  static_assert(
      std::is_same_v<std::decay_t<decltype(std::declval<complex_t>().x)>,
                     std::decay_t<decltype(std::declval<fftw_complex_t>()[0])>>,
      "Complex type has the wrong precision for FFTW");

  static constexpr auto plan_many_dft_r2c = []() {
    if constexpr (std::is_same_v<real_t, double>) {
      return fftw_plan_many_dft_r2c;
    } else {
      return fftwf_plan_many_dft_r2c;
    }
  }();
  static constexpr auto plan_many_dft_c2r = []() {
    if constexpr (std::is_same_v<real_t, double>) {
      return fftw_plan_many_dft_c2r;
    } else {
      return fftwf_plan_many_dft_c2r;
    }
  }();
  static constexpr auto destroy_plan = []() {
    if constexpr (std::is_same_v<real_t, double>) {
      return fftw_destroy_plan;
    } else {
      return fftwf_destroy_plan;
    }
  }();
  static constexpr auto execute = []() {
    if constexpr (std::is_same_v<real_t, double>) {
      return fftw_execute;
    } else {
      return fftwf_execute;
    }
  }();

public:
  static constexpr int dim_v = Dim;

  FFTWFFT(const zisa::array_view<complex_t, dim_v + 1> &u_hat,
          const zisa::array_view<real_t, dim_v + 1> &u,
          int direction = FFT_FORWARD | FFT_BACKWARD)
      : super(direction) {
    initialize(u_hat, u);
  }

  virtual ~FFTWFFT() override {
    if (direction_ & FFT_FORWARD) {
      destroy_plan(plan_forward_);
    }
    if (direction_ & FFT_BACKWARD) {
      destroy_plan(plan_backward_);
    }
  }

  using super::initialize;

  virtual void forward() override {
    LOG_ERR_IF((direction_ & FFT_FORWARD) == 0,
               "Forward operation was not initialized");
    AZEBAN_PROFILE_START("FFTWFFT::forward");
    execute(plan_forward_);
    AZEBAN_PROFILE_STOP("FFTWFFT::forward");
  }

  virtual void backward() override {
    LOG_ERR_IF((direction_ & FFT_BACKWARD) == 0,
               "Backward operation was not initialized");
    AZEBAN_PROFILE_START("FFTWFFT::backward");
    execute(plan_backward_);
    AZEBAN_PROFILE_STOP("FFTWFFT::backward");
  }

protected:
  using super::data_dim_;
  using super::direction_;
  using super::u_;
  using super::u_hat_;

  virtual void
  do_initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat,
                const zisa::array_view<real_t, Dim + 1> &u) override {
    assert(u_hat.memory_location() == zisa::device_type::cpu
           && "FFTW is CPU only!");
    assert(u.memory_location() == zisa::device_type::cpu
           && "FFTW is CPU only!");
    int rdist = 1;
    int cdist = 1;
    int n[dim_v];
    for (int i = 0; i < dim_v; ++i) {
      rdist *= u.shape()[i + 1];
      cdist *= u_hat.shape()[i + 1];
      n[i] = u.shape()[i + 1];
    }
    // Create a plan for the forward operation
    if (direction_ & FFT_FORWARD) {
      plan_forward_ = plan_many_dft_r2c(
          dim_v,                                       // rank
          n,                                           // n
          data_dim_,                                   // howmany
          u.raw(),                                     // in
          NULL,                                        // inembed
          1,                                           // istride
          rdist,                                       // idist
          reinterpret_cast<real_t(*)[2]>(u_hat.raw()), // out
          NULL,                                        // onembed
          1,                                           // ostride
          cdist,                                       // odist
          FFTW_MEASURE);                               // flags
    }
    if (direction_ & FFT_BACKWARD) {
      // Create a plan for the backward operation
      plan_backward_
          = plan_many_dft_c2r(dim_v,     // rank
                              n,         // n
                              data_dim_, // howmany
                              reinterpret_cast<real_t(*)[2]>(u_hat.raw()), // in
                              NULL,          // inembed
                              1,             // istride
                              cdist,         // idist
                              u.raw(),       // out
                              NULL,          // onembed
                              1,             // ostride
                              rdist,         // odist
                              FFTW_MEASURE); // flags
    }
  }

private:
  fftw_plan_t plan_forward_;
  fftw_plan_t plan_backward_;
};

}

#endif
