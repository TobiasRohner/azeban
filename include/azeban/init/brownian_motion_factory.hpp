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
#ifndef BROWNIAN_MOTION_FACTORY_H_
#define BROWNIAN_MOTION_FACTORY_H_

#include "brownian_motion.hpp"
#include <azeban/random/random_variable_factory.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>>
make_brownian_motion(const nlohmann::json &config, RNG &rng) {
  RandomVariable<real_t> H(std::make_shared<Delta<real_t>>(0.5));
  if (config.contains("hurst")) {
    H = make_random_variable<real_t>(config["hurst"], rng);
  }
  return std::make_shared<BrownianMotion<Dim>>(H, rng);
}

}

#endif
