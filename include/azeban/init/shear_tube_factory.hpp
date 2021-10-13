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
#ifndef SHEAR_TUBE_FACTORY_H_
#define SHEAR_TUBE_FACTORY_H_

#include "shear_tube.hpp"
#include <azeban/random/random_variable_factory.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>> make_shear_tube(const nlohmann::json &config,
                                                  RNG &rng) {
  if constexpr (Dim == 3) {
    if (!config.contains("rho")) {
      fmt::print(stderr,
                 "Shear Tube initialization is missing parameter \"rho\"\n");
      exit(1);
    }
    if (!config.contains("delta")) {
      fmt::print(stderr,
                 "Shear Tube initialization is missing parameter "
                 "\"delta\"\n");
      exit(1);
    }
    RandomVariable<real_t> rho
        = make_random_variable<real_t>(config["rho"], rng);
    RandomVariable<real_t> delta
        = make_random_variable<real_t>(config["delta"], rng);
    return std::make_shared<ShearTube>(rho, delta);
  } else {
    fmt::print(stderr, "Shear Tube is only available for 3D simulations\n");
    exit(1);
  }
}

}

#endif
