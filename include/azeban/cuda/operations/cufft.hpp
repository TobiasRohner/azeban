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
#ifndef AZEBAN_CUDA_OPERATIONS_CUFFT_HPP_
#define AZEBAN_CUDA_OPERATIONS_CUFFT_HPP_

#include <azeban/operations/fft.hpp>
#include <cufft.h>

namespace azeban {

template <int Dim>
class CUFFT_R2C final : public FFT<Dim, real_t> {
  using super = FFT<Dim, real_t>;

  static_assert(std::is_same_v<real_t, float> || std::is_same_v<real_t, double>,
                "Only single and double precision are supported by CUFFT_R2C");

public:
  static constexpr int dim_v = Dim;

  template <bool enable = Dim == 1, typename = std::enable_if_t<enable>>
  CUFFT_R2C(int direction = FFT_FORWARD | FFT_BACKWARD,
            bool transform_x = true);
  template <bool enable = Dim == 2, typename = std::enable_if_t<enable>>
  CUFFT_R2C(int direction = FFT_FORWARD | FFT_BACKWARD,
            bool transform_x = true,
            bool transform_y = true);
  template <bool enable = Dim == 3, typename = std::enable_if_t<enable>>
  CUFFT_R2C(int direction = FFT_FORWARD | FFT_BACKWARD,
            bool transform_x = true,
            bool transform_y = true,
            bool transform_z = true);
  template <bool enable = Dim == 1, typename = std::enable_if_t<enable>>
  CUFFT_R2C(const zisa::array_view<complex_t, Dim + 1> &u_hat,
            const zisa::array_view<real_t, Dim + 1> &u,
            int direction = FFT_FORWARD | FFT_BACKWARD,
            bool transform_x = true);
  template <bool enable = Dim == 2, typename = std::enable_if_t<enable>>
  CUFFT_R2C(const zisa::array_view<complex_t, Dim + 1> &u_hat,
            const zisa::array_view<real_t, Dim + 1> &u,
            int direction = FFT_FORWARD | FFT_BACKWARD,
            bool transform_x = true,
            bool transform_y = true);
  template <bool enable = Dim == 3, typename = std::enable_if_t<enable>>
  CUFFT_R2C(const zisa::array_view<complex_t, Dim + 1> &u_hat,
            const zisa::array_view<real_t, Dim + 1> &u,
            int direction = FFT_FORWARD | FFT_BACKWARD,
            bool transform_x = true,
            bool transform_y = true,
            bool transform_z = true);
  virtual ~CUFFT_R2C() override;
  using super::initialize;
  virtual size_t get_work_area_size() const override;
  using super::set_work_area;
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

  virtual void do_initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat,
                             const zisa::array_view<real_t, Dim + 1> &u,
                             bool allocate_work_area) override;
  virtual void do_set_work_area(void *work_area) override;

private:
  cufftHandle plan_forward_;
  cufftHandle plan_backward_;
  bool custom_work_area_ = false;
  size_t work_area_size_ = 0;
  void *work_area_ = nullptr;

  static constexpr cufftType type_forward
      = std::is_same_v<float, real_t> ? ::CUFFT_R2C : ::CUFFT_D2Z;
  static constexpr cufftType type_backward
      = std::is_same_v<float, real_t> ? ::CUFFT_C2R : ::CUFFT_Z2D;
};

template <int Dim>
class CUFFT_C2C final : public FFT<Dim, complex_t> {
  using super = FFT<Dim, complex_t>;

  static_assert(std::is_same_v<real_t, float> || std::is_same_v<real_t, double>,
                "Only single and double precision are supported by CUFFT_R2C");

public:
  static constexpr int dim_v = Dim;

  template <bool enable = Dim == 1, typename = std::enable_if_t<enable>>
  CUFFT_C2C(int direction = FFT_FORWARD | FFT_BACKWARD,
            bool transform_x = true);
  template <bool enable = Dim == 2, typename = std::enable_if_t<enable>>
  CUFFT_C2C(int direction = FFT_FORWARD | FFT_BACKWARD,
            bool transform_x = true,
            bool transform_y = true);
  template <bool enable = Dim == 3, typename = std::enable_if_t<enable>>
  CUFFT_C2C(int direction = FFT_FORWARD | FFT_BACKWARD,
            bool transform_x = true,
            bool transform_y = true,
            bool transform_z = true);
  template <bool enable = Dim == 1, typename = std::enable_if_t<enable>>
  CUFFT_C2C(const zisa::array_view<complex_t, Dim + 1> &u_hat,
            const zisa::array_view<complex_t, Dim + 1> &u,
            int direction = FFT_FORWARD | FFT_BACKWARD,
            bool transform_x = true);
  template <bool enable = Dim == 2, typename = std::enable_if_t<enable>>
  CUFFT_C2C(const zisa::array_view<complex_t, Dim + 1> &u_hat,
            const zisa::array_view<complex_t, Dim + 1> &u,
            int direction = FFT_FORWARD | FFT_BACKWARD,
            bool transform_x = true,
            bool transform_y = true);
  template <bool enable = Dim == 3, typename = std::enable_if_t<enable>>
  CUFFT_C2C(const zisa::array_view<complex_t, Dim + 1> &u_hat,
            const zisa::array_view<complex_t, Dim + 1> &u,
            int direction = FFT_FORWARD | FFT_BACKWARD,
            bool transform_x = true,
            bool transform_y = true,
            bool transform_z = true);
  virtual ~CUFFT_C2C() override;
  using super::initialize;
  virtual size_t get_work_area_size() const override;
  using super::set_work_area;
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

  virtual void do_initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat,
                             const zisa::array_view<complex_t, Dim + 1> &u,
                             bool allocate_work_area) override;
  virtual void do_set_work_area(void *work_area) override;

private:
  cufftHandle plan_forward_;
  cufftHandle plan_backward_;
  bool custom_work_area_ = false;
  size_t work_area_size_ = 0;
  void *work_area_ = nullptr;

  static constexpr cufftType type_forward
      = std::is_same_v<float, real_t> ? ::CUFFT_C2C : ::CUFFT_Z2Z;
  static constexpr cufftType type_backward
      = std::is_same_v<float, real_t> ? ::CUFFT_C2C : ::CUFFT_Z2Z;
};

namespace detail {

template <int Dim, typename ScalarU>
struct CUFFT_Impl {
  static_assert(!std::is_same_v<ScalarU, ScalarU>,
                "Unsupported scalar value for u");
};

template <int Dim>
struct CUFFT_Impl<Dim, real_t> {
  using type = CUFFT_R2C<Dim>;
};

template <int Dim>
struct CUFFT_Impl<Dim, complex_t> {
  using type = CUFFT_C2C<Dim>;
};

}

template <int Dim, typename ScalarU = real_t>
using CUFFT = typename detail::CUFFT_Impl<Dim, ScalarU>::type;

}

#endif
