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
#include <azeban/init/shear_tube.hpp>
#include <azeban/operations/fft.hpp>
#include <fmt/core.h>
#include <zisa/memory/array.hpp>

namespace azeban {

void ShearTube::do_initialize(const zisa::array_view<real_t, 4> &u) {
  const auto init = [&](auto &&u_) {
    const zisa::int_t N = u_.shape(1);
    const real_t rho = rho_.get();
    const real_t delta = delta_.get();
    for (zisa::int_t i = 0; i < N; ++i) {
      for (zisa::int_t j = 0; j < N; ++j) {
        for (zisa::int_t k = 0; k < N; ++k) {
          const real_t x = static_cast<real_t>(i) / N;
          const real_t y = static_cast<real_t>(j) / N;
          const real_t z = static_cast<real_t>(k) / N;
          const real_t r
              = zisa::sqrt(zisa::pow<2>(y - 0.5) + zisa::pow<2>(z - 0.5));
          u_(0, i, j, k) = std::tanh(2 * zisa::pi * (r - 0.25) / rho);
          u_(1, i, j, k) = delta * zisa::sin(2 * zisa::pi * x);
          u_(2, i, j, k) = 0;
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
    LOG_ERR("Unknown Memory Location");
  }
}

void ShearTube::do_initialize(const zisa::array_view<complex_t, 4> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 4>(zisa::shape_t<4>(3, N, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<3>(u_hat, u);
  initialize(u);
  fft->forward();
}

}
