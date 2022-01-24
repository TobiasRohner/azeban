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
#ifndef FFT_BASE_H_WIQBB
#define FFT_BASE_H_WIQBB

#include <azeban/config.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

enum fft_direction { FFT_FORWARD = 1, FFT_BACKWARD = 2 };

template <int Dim>
class FFT {
public:
  static constexpr int dim_v = Dim;

  FFT(const zisa::array_view<complex_t, dim_v + 1> &u_hat,
      const zisa::array_view<real_t, dim_v + 1> &u,
      int direction)
      : u_hat_(u_hat), u_(u), data_dim_(u_.shape()[0]), direction_(direction) {
    assert(u_hat.shape()[0] == u.shape()[0]
           && "Dimensionality of data elements must be equal!");
  }

  FFT() = default;
  FFT(const FFT &) = default;
  FFT(FFT &&) = default;

  virtual ~FFT() = default;

  FFT &operator=(const FFT &) = default;
  FFT &operator=(FFT &&) = default;

  virtual void forward() = 0;
  virtual void backward() = 0;

  decltype(auto) shape() const { return u_.shape(); }
  decltype(auto) shape(zisa::int_t i) const { return u_.shape(i); }

  const zisa::array_view<complex_t, dim_v + 1> &u_hat() { return u_hat_; }
  const zisa::array_const_view<complex_t, dim_v + 1> u_hat() const {
    return u_hat_;
  }
  const zisa::array_view<real_t, dim_v + 1> &u() { return u_; }
  const zisa::array_const_view<real_t, dim_v + 1> u() const { return u_; }

  virtual void *get_work_area() const { return nullptr; }

protected:
  zisa::array_view<complex_t, dim_v + 1> u_hat_;
  zisa::array_view<real_t, dim_v + 1> u_;
  zisa::int_t data_dim_;
  int direction_;
};

}

#endif
