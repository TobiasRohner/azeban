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
#ifndef SPHERE_FACTORY_H_
#define SPHERE_FACTORY_H_

#include "sphere.hpp"
#include <fmt/core.h>
#include <vector>

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>> make_sphere(const nlohmann::json &config) {
  if constexpr (Dim == 2 || Dim == 3) {
    if (!config.contains("center")) {
      fmt::print(stderr, "Sphere initializer needs parameter \"center\"\n");
      exit(1);
    }
    if (!config.contains("radius")) {
      fmt::print(stderr, "Sphere initializer needs parameter \"radius\"\n");
      exit(1);
    }
    const std::vector<real_t> center
        = config["center"].get<std::vector<real_t>>();
    const real_t radius = config["radius"];
    if (center.size() != Dim) {
      fmt::print(stderr,
                 "Center of sphere initializer must have {} components instead "
                 "of {}\n",
                 Dim,
                 center.size());
      exit(1);
    }
    if constexpr (Dim == 2) {
      return std::make_shared<Sphere<Dim>>(
          std::array<real_t, 2>{center[0], center[1]}, radius);
    } else {
      return std::make_shared<Sphere<Dim>>(
          std::array<real_t, 3>{center[0], center[1], center[2]}, radius);
    }
  } else {
    fmt::print(stderr, "Sphere initializer is only defined for 2D or 3D\n");
    exit(1);
  }
}

}

#endif
