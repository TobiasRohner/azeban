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
#include <azeban/init/taylor_green.hpp>
#include <azeban/operations/fft.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

void TaylorGreen<2>::do_initialize(const zisa::array_view<real_t, 3> &u) {
  const auto init = [&](auto &&u_) {
    const real_t A = 1;
    const real_t B = -1;
    const zisa::int_t N = u_.shape(1);
    real_t deltas[8];
    for (int i = 0; i < 8; ++i) {
      deltas[i] = delta_.get() / 4;
    }
    for (zisa::int_t i = 0; i < N; ++i) {
      for (zisa::int_t j = 0; j < N; ++j) {
        const real_t x = 2 * zisa::pi / N * i;
        const real_t y = 2 * zisa::pi / N * j;
        u_(0, i, j) = A * zisa::cos(x) * zisa::sin(y);
        u_(1, i, j) = B * zisa::sin(x) * zisa::cos(y);
        for (int d = 0; d < 2; ++d) {
          u_(d, i, j)
              += deltas[0 + 4 * d] * zisa::sin(2 * x) * zisa::sin(2 * y);
          u_(d, i, j)
              += deltas[1 + 4 * d] * zisa::sin(2 * x) * zisa::cos(2 * y);
          u_(d, i, j)
              += deltas[2 + 4 * d] * zisa::cos(2 * x) * zisa::sin(2 * y);
          u_(d, i, j)
              += deltas[3 + 4 * d] * zisa::cos(2 * x) * zisa::cos(2 * y);
        }
      }
    }
  };
  if (u.memory_location() == zisa::device_type::cpu) {
    init(u);
  } else if (u.memory_location() == zisa::device_type::cuda) {
    auto h_u = zisa::array<real_t, 3>(u.shape(), zisa::device_type::cpu);
    init(h_u);
    zisa::copy(u, h_u);
  } else {
    LOG_ERR("Unsupported Memory Location");
  }
}

void TaylorGreen<2>::do_initialize(
    const zisa::array_view<complex_t, 3> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 3>(zisa::shape_t<3>(2, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<2>(u_hat, u);
  do_initialize(u);
  fft->forward();
}

void TaylorGreen<3>::do_initialize(const zisa::array_view<real_t, 4> &u) {
  const auto init = [&](auto &&u_) {
    const real_t A = 1;
    const real_t B = -1;
    const real_t C = 0;
    const zisa::int_t N = u_.shape(1);
    real_t deltas[24];
    for (int i = 0; i < 24; ++i) {
      deltas[i] = delta_.get() / 8;
    }
    for (zisa::int_t i = 0; i < N; ++i) {
      for (zisa::int_t j = 0; j < N; ++j) {
        for (zisa::int_t k = 0; k < N; ++k) {
          const real_t x = 2 * zisa::pi / N * i;
          const real_t y = 2 * zisa::pi / N * j;
          const real_t z = 2 * zisa::pi / N * k;
          u_(0, i, j, k) = A * zisa::cos(x) * zisa::sin(y) * zisa::sin(z);
          u_(1, i, j, k) = B * zisa::sin(x) * zisa::cos(y) * zisa::sin(z);
          u_(2, i, j, k) = C * zisa::sin(x) * zisa::sin(y) * zisa::cos(z);
          for (int d = 0; d < 3; ++d) {
            u_(d, i, j, k) += deltas[0 + 8 * d] * zisa::sin(2 * x)
                              * zisa::sin(2 * y) * zisa::sin(2 * z);
            u_(d, i, j, k) += deltas[1 + 8 * d] * zisa::sin(2 * x)
                              * zisa::sin(2 * y) * zisa::cos(2 * z);
            u_(d, i, j, k) += deltas[2 + 8 * d] * zisa::sin(2 * x)
                              * zisa::cos(2 * y) * zisa::sin(2 * z);
            u_(d, i, j, k) += deltas[3 + 8 * d] * zisa::sin(2 * x)
                              * zisa::cos(2 * y) * zisa::cos(2 * z);
            u_(d, i, j, k) += deltas[4 + 8 * d] * zisa::cos(2 * x)
                              * zisa::sin(2 * y) * zisa::sin(2 * z);
            u_(d, i, j, k) += deltas[5 + 8 * d] * zisa::cos(2 * x)
                              * zisa::sin(2 * y) * zisa::cos(2 * z);
            u_(d, i, j, k) += deltas[6 + 8 * d] * zisa::cos(2 * x)
                              * zisa::cos(2 * y) * zisa::sin(2 * z);
            u_(d, i, j, k) += deltas[7 + 8 * d] * zisa::cos(2 * x)
                              * zisa::cos(2 * y) * zisa::cos(2 * z);
          }
        }
      }
    }
  };
  if (u.memory_location() == zisa::device_type::cpu) {
    init(u);
  } else if (u.memory_location() == zisa::device_type::cuda) {
    auto h_u = zisa::array<real_t, 4>(u.shape(), zisa::device_type::cpu);
    init(h_u);
    zisa::copy(u, h_u);
  } else {
    LOG_ERR("Unsupported Memory Location");
  }
}

void TaylorGreen<3>::do_initialize(
    const zisa::array_view<complex_t, 4> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 4>(zisa::shape_t<4>(3, N, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<3>(u_hat, u);
  do_initialize(u);
  fft->forward();
}

}
