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
#ifndef AZEBAN_OPERATIONS_FFTWFFT_HPP_
#define AZEBAN_OPERATIONS_FFTWFFT_HPP_

#include <azeban/operations/fft.hpp>
#include <fftw3.h>

namespace azeban {

namespace detail {

template <typename Scalar>
struct fftw_complex_impl;

template <>
struct fftw_complex_impl<float> {
  using type = fftwf_complex;
};

template <>
struct fftw_complex_impl<double> {
  using type = fftw_complex;
};

template <typename Scalar>
struct fftw_plan_impl;

template <>
struct fftw_plan_impl<float> {
  using type = fftwf_plan;
};

template <>
struct fftw_plan_impl<double> {
  using type = fftw_plan;
};

}

using fftw_complex_t = typename detail::fftw_complex_impl<real_t>::type;
using fftw_plan_t = typename detail::fftw_plan_impl<real_t>::type;

template <int Dim>
class FFTWFFT_R2C final : public FFT<Dim, real_t> {
  using super = FFT<Dim, real_t>;

  static_assert(
      std::is_same_v<real_t,
                     std::decay_t<decltype(std::declval<fftw_complex_t>()[0])>>,
      "Real type has the wrong precision for FFTW");
  static_assert(
      std::is_same_v<std::decay_t<decltype(std::declval<complex_t>().x)>,
                     std::decay_t<decltype(std::declval<fftw_complex_t>()[0])>>,
      "Complex type has the wrong precision for FFTW");

public:
  static constexpr int dim_v = Dim;

  template <bool enable = Dim == 1, typename = std::enable_if_t<enable>>
  FFTWFFT_R2C(int direction = FFT_FORWARD | FFT_BACKWARD,
              bool transform_x = true);
  template <bool enable = Dim == 2, typename = std::enable_if_t<enable>>
  FFTWFFT_R2C(int direction = FFT_FORWARD | FFT_BACKWARD,
              bool transform_x = true,
              bool transform_y = true);
  template <bool enable = Dim == 3, typename = std::enable_if_t<enable>>
  FFTWFFT_R2C(int direction = FFT_FORWARD | FFT_BACKWARD,
              bool transform_x = true,
              bool transform_y = true,
              bool transform_z = true);
  template <bool enable = Dim == 1, typename = std::enable_if_t<enable>>
  FFTWFFT_R2C(const zisa::array_view<complex_t, Dim + 1> &u_hat,
              const zisa::array_view<real_t, Dim + 1> &u,
              int direction = FFT_FORWARD | FFT_BACKWARD,
              bool transform_x = true);
  template <bool enable = Dim == 2, typename = std::enable_if_t<enable>>
  FFTWFFT_R2C(const zisa::array_view<complex_t, Dim + 1> &u_hat,
              const zisa::array_view<real_t, Dim + 1> &u,
              int direction = FFT_FORWARD | FFT_BACKWARD,
              bool transform_x = true,
              bool transform_y = true);
  template <bool enable = Dim == 3, typename = std::enable_if_t<enable>>
  FFTWFFT_R2C(const zisa::array_view<complex_t, Dim + 1> &u_hat,
              const zisa::array_view<real_t, Dim + 1> &u,
              int direction = FFT_FORWARD | FFT_BACKWARD,
              bool transform_x = true,
              bool transform_y = true,
              bool transform_z = true);
  virtual ~FFTWFFT_R2C() override;
  using super::initialize;
  virtual void forward() override;
  virtual void backward() override;
  using super::is_backward;
  using super::is_forward;

protected:
  using super::data_dim_;
  using super::direction_;
  using super::transform_dims_;
  using super::u_;
  using super::u_hat_;

  virtual void
  do_initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat,
                const zisa::array_view<real_t, Dim + 1> &u) override;

private:
  fftw_plan_t plan_forward_;
  fftw_plan_t plan_backward_;
};

template <int Dim>
class FFTWFFT_C2C final : public FFT<Dim, complex_t> {
  using super = FFT<Dim, complex_t>;

  static_assert(
      std::is_same_v<real_t,
                     std::decay_t<decltype(std::declval<fftw_complex_t>()[0])>>,
      "Real type has the wrong precision for FFTW");
  static_assert(
      std::is_same_v<std::decay_t<decltype(std::declval<complex_t>().x)>,
                     std::decay_t<decltype(std::declval<fftw_complex_t>()[0])>>,
      "Complex type has the wrong precision for FFTW");

public:
  static constexpr int dim_v = Dim;

  template <bool enable = Dim == 1, typename = std::enable_if_t<enable>>
  FFTWFFT_C2C(int direction = FFT_FORWARD | FFT_BACKWARD,
              bool transform_x = true);
  template <bool enable = Dim == 2, typename = std::enable_if_t<enable>>
  FFTWFFT_C2C(int direction = FFT_FORWARD | FFT_BACKWARD,
              bool transform_x = true,
              bool transform_y = true);
  template <bool enable = Dim == 3, typename = std::enable_if_t<enable>>
  FFTWFFT_C2C(int direction = FFT_FORWARD | FFT_BACKWARD,
              bool transform_x = true,
              bool transform_y = true,
              bool transform_z = true);
  template <bool enable = Dim == 1, typename = std::enable_if_t<enable>>
  FFTWFFT_C2C(const zisa::array_view<complex_t, Dim + 1> &u_hat,
              const zisa::array_view<complex_t, Dim + 1> &u,
              int direction = FFT_FORWARD | FFT_BACKWARD,
              bool transform_x = true);
  template <bool enable = Dim == 2, typename = std::enable_if_t<enable>>
  FFTWFFT_C2C(const zisa::array_view<complex_t, Dim + 1> &u_hat,
              const zisa::array_view<complex_t, Dim + 1> &u,
              int direction = FFT_FORWARD | FFT_BACKWARD,
              bool transform_x = true,
              bool transform_y = true);
  template <bool enable = Dim == 3, typename = std::enable_if_t<enable>>
  FFTWFFT_C2C(const zisa::array_view<complex_t, Dim + 1> &u_hat,
              const zisa::array_view<complex_t, Dim + 1> &u,
              int direction = FFT_FORWARD | FFT_BACKWARD,
              bool transform_x = true,
              bool transform_y = true,
              bool transform_z = true);
  virtual ~FFTWFFT_C2C() override;
  using super::initialize;
  virtual void forward() override;
  virtual void backward() override;
  using super::is_backward;
  using super::is_forward;

protected:
  using super::data_dim_;
  using super::direction_;
  using super::transform_dims_;
  using super::u_;
  using super::u_hat_;

  virtual void
  do_initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat,
                const zisa::array_view<complex_t, Dim + 1> &u) override;

private:
  fftw_plan_t plan_forward_;
  fftw_plan_t plan_backward_;
};

namespace detail {

template <int Dim, typename ScalarU>
struct FFTWFFT_Impl {
  static_assert(!std::is_same_v<ScalarU, ScalarU>,
                "Unsupported scalar value for u");
};

template <int Dim>
struct FFTWFFT_Impl<Dim, real_t> {
  using type = FFTWFFT_R2C<Dim>;
};

template <int Dim>
struct FFTWFFT_Impl<Dim, complex_t> {
  using type = FFTWFFT_C2C<Dim>;
};

}

template <int Dim, typename ScalarU = real_t>
using FFTWFFT = typename detail::FFTWFFT_Impl<Dim, ScalarU>::type;

}

#endif
