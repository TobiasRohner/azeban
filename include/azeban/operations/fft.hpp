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
#ifndef AZEBAN_OPERATIONS_FFT_HPP_
#define AZEBAN_OPERATIONS_FFT_HPP_

#include <azeban/config.hpp>
#include <string>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

enum fft_direction { FFT_FORWARD = 1, FFT_BACKWARD = 2 };

// If no benchmark file exists, leave the filename empty
zisa::int_t optimal_fft_size(const std::string &benchmark_file,
                             zisa::int_t N,
                             int dim,
                             int n_vars,
                             zisa::device_type device);

template <int Dim, typename ScalarU = real_t>
class FFT {
public:
  static constexpr int dim_v = Dim;
  using scalar_u_t = ScalarU;
  static_assert(std::is_same_v<scalar_u_t,
                               real_t> || std::is_same_v<scalar_u_t, complex_t>,
                "ScalarU must be eiter real_t or complex_t");

  template <bool enable = Dim == 1, typename = std::enable_if_t<enable>>
  FFT(int direction, bool transform_x = true)
      : direction_(direction),
        transform_dims_{transform_x},
        u_hat_({}, nullptr),
        u_({}, nullptr),
        data_dim_(0) {}
  template <bool enable = Dim == 2, typename = std::enable_if_t<enable>>
  FFT(int direction, bool transform_x = true, bool transform_y = true)
      : direction_(direction),
        transform_dims_{transform_x, transform_y},
        u_hat_({}, nullptr),
        u_({}, nullptr),
        data_dim_(0) {}
  template <bool enable = Dim == 3, typename = std::enable_if_t<enable>>
  FFT(int direction,
      bool transform_x = true,
      bool transform_y = true,
      bool transform_z = true)
      : direction_(direction),
        transform_dims_{transform_x, transform_y, transform_z},
        u_hat_({}, nullptr),
        u_({}, nullptr),
        data_dim_(0) {}

  FFT() = default;
  FFT(const FFT &) = default;
  FFT(FFT &&) = default;

  virtual ~FFT() = default;

  FFT &operator=(const FFT &) = default;
  FFT &operator=(FFT &&) = default;

  void initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat,
                  const zisa::array_view<scalar_u_t, Dim + 1> &u);

  virtual void forward() = 0;
  virtual void backward() = 0;

  bool is_forward() const { return direction_ & FFT_FORWARD; }
  bool is_backward() const { return direction_ & FFT_BACKWARD; }

  decltype(auto) shape() const { return u_.shape(); }
  decltype(auto) shape(zisa::int_t i) const { return u_.shape(i); }

  const zisa::array_view<complex_t, dim_v + 1> &u_hat() { return u_hat_; }
  const zisa::array_const_view<complex_t, dim_v + 1> u_hat() const {
    return u_hat_;
  }
  const zisa::array_view<scalar_u_t, dim_v + 1> &u() { return u_; }
  const zisa::array_const_view<scalar_u_t, dim_v + 1> u() const { return u_; }

  zisa::shape_t<Dim + 1>
  output_shape(const zisa::shape_t<Dim + 1> &input_shape) const;

  virtual void *get_work_area() const { return nullptr; }

protected:
  int direction_;
  bool transform_dims_[dim_v]; // TODO: Initialize this
  zisa::array_view<complex_t, Dim + 1> u_hat_;
  zisa::array_view<scalar_u_t, Dim + 1> u_;
  int data_dim_;

  virtual void do_initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat,
                             const zisa::array_view<scalar_u_t, Dim + 1> &u)
      = 0;
};

}

#endif
