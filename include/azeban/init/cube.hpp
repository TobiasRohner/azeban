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
#ifndef AZEBAN_INIT_CUBE_HPP_
#define AZEBAN_INIT_CUBE_HPP_

#include "initializer.hpp"
#include <array>

namespace azeban {

template <int Dim>
class Cube final : public Initializer<Dim> {
  using super = Initializer<Dim>;
  static constexpr int dim_v = Dim;

public:
  Cube(const std::array<real_t, dim_v> &corner, std::array<real_t, dim_v> size)
      : corner_(corner), size_(size) {}
  Cube(const Cube &) = default;
  Cube(Cube &&) = default;

  virtual ~Cube() override = default;

  Cube &operator=(const Cube &) = default;
  Cube &operator=(Cube &) = default;

  virtual void
  initialize(const zisa::array_view<real_t, dim_v + 1> &u) override {
    const auto init = [&](auto &&u_) {
      const zisa::int_t N = u_.shape(1);
      if constexpr (dim_v == 2) {
        for (zisa::int_t i = 0; i < N; ++i) {
          for (zisa::int_t j = 0; j < N; ++j) {
            const real_t x = static_cast<real_t>(i) / N;
            const real_t y = static_cast<real_t>(j) / N;
            u_(0, i, j) = x >= corner_[0] && x <= corner_[0] + size_[0]
                                  && y >= corner_[1]
                                  && y <= corner_[1] + size_[1]
                              ? 1
                              : 0;
          }
        }
      } else if constexpr (dim_v == 3) {
        for (zisa::int_t i = 0; i < N; ++i) {
          for (zisa::int_t j = 0; j < N; ++j) {
            for (zisa::int_t k = 0; k < N; ++k) {
              const real_t x = static_cast<real_t>(i) / N;
              const real_t y = static_cast<real_t>(j) / N;
              const real_t z = static_cast<real_t>(k) / N;
              u_(0, i, j, k)
                  = x >= corner_[0] && x <= corner_[0] + size_[0]
                            && y >= corner_[1] && y <= corner_[1] + size_[1]
                            && z >= corner_[2] && z <= corner_[2] + size_[2]
                        ? 1
                        : 0;
            }
          }
        }
      } else {
        LOG_ERR("Unsupported Dimension");
      }
    };
    if (u.memory_location() == zisa::device_type::cpu) {
      init(u);
    } else if (u.memory_location() == zisa::device_type::cuda) {
      auto h_u
          = zisa::array<real_t, dim_v + 1>(u.shape(), zisa::device_type::cpu);
      init(h_u);
      zisa::copy(u, h_u);
    } else {
      LOG_ERR("Unknown Memory Location");
    }
  }

  virtual void
  initialize(const zisa::array_view<complex_t, dim_v + 1> &u_hat) override {
    const zisa::int_t N = u_hat.shape(1);
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = 1;
    for (int i = 0; i < dim_v; ++i) {
      shape[i + 1] = N;
    }
    auto u = zisa::array<real_t, dim_v + 1>(shape, u_hat.memory_location());
    auto fft = make_fft<dim_v>(u_hat, u);
    initialize(u);
    fft->forward();
  }

protected:
  virtual void
  do_initialize(const zisa::array_view<real_t, dim_v + 1> &) override {}
  virtual void
  do_initialize(const zisa::array_view<complex_t, dim_v + 1> &) override {}

private:
  std::array<real_t, dim_v> corner_;
  std::array<real_t, dim_v> size_;
};

}

#endif
