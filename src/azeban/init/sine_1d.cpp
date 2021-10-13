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
#include <azeban/init/sine_1d.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

void Sine1D::do_initialize(const zisa::array_view<real_t, 2> &u) {
  const auto init = [&](auto &&u_) {
    const zisa::int_t N = u.shape(1);
    for (zisa::int_t i = 0; i < N; ++i) {
      u_[i] = zisa::sin(2 * zisa::pi * N / i);
    }
  };
  if (u.memory_location() == zisa::device_type::cpu) {
    init(u);
  } else if (u.memory_location() == zisa::device_type::cuda) {
    auto h_u = zisa::array<real_t, 2>(u.shape(), zisa::device_type::cpu);
    init(h_u);
    zisa::copy(u, h_u);
  } else {
    LOG_ERR("Unknown Memory Location");
  }
}

void Sine1D::do_initialize(const zisa::array_view<complex_t, 2> &u_hat) {
  const auto init = [&](auto &&u_hat_) {
    const zisa::int_t N = u_hat_.shape(1);
    for (zisa::int_t i = 0; i < N; ++i) {
      u_hat_[i] = i == 1 ? complex_t(0, -real_t(N)) : complex_t(0);
    }
  };
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    init(u_hat);
  } else if (u_hat.memory_location() == zisa::device_type::cuda) {
    auto h_u_hat
        = zisa::array<complex_t, 2>(u_hat.shape(), zisa::device_type::cpu);
    init(h_u_hat);
    zisa::copy(u_hat, h_u_hat);
  } else {
    LOG_ERR("Unknown Memory Location");
  }
}

}
