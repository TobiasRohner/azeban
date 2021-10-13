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
#include <azeban/init/taylor_vortex.hpp>
#include <azeban/operations/fft.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

void TaylorVortex::initialize(const zisa::array_view<real_t, 3> &u) {
  do_initialize(u);
}

void TaylorVortex::do_initialize(const zisa::array_view<real_t, 3> &u) {
  const auto init = [&](auto &&u_) {
    const zisa::int_t N = u_.shape(1);
    for (zisa::int_t i = 0; i < N; ++i) {
      for (zisa::int_t j = 0; j < N; ++j) {
        const real_t x = 16 * static_cast<real_t>(i) / N - 8;
        const real_t y = 16 * static_cast<real_t>(j) / N - 8;
        u_(0, i, j)
            = (-y * zisa::exp(0.5 * (1 - zisa::pow<2>(x) - zisa::pow<2>(y)))
               + 8)
              / 16;
        u_(1, i, j)
            = x * zisa::exp(0.5 * (1 - zisa::pow<2>(x) - zisa::pow<2>(y))) / 16;
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
    LOG_ERR("Unknown Memory Location");
  }
}

void TaylorVortex::initialize(const zisa::array_view<complex_t, 3> &u_hat) {
  do_initialize(u_hat);
}

void TaylorVortex::do_initialize(const zisa::array_view<complex_t, 3> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 3>(zisa::shape_t<3>(2, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<2>(u_hat, u);
  initialize(u);
  fft->forward();
}

}
