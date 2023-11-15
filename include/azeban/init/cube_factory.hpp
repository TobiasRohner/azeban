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
#ifndef AZEBAN_INIT_CUBE_FACTORY_HPP_
#define AZEBAN_INIT_CUBE_FACTORY_HPP_

#include "cube.hpp"
#include <fmt/core.h>
#include <vector>

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>> make_cube(const nlohmann::json &config) {
  if constexpr (Dim == 2 || Dim == 3) {
    if (!config.contains("corner")) {
      fmt::print(stderr, "Cube initializer needs parameter \"corner\"\n");
      exit(1);
    }
    if (!config.contains("size")) {
      fmt::print(stderr, "Cube initializer needs parameter \"size\"\n");
      exit(1);
    }
    const std::vector<real_t> corner
        = config["corner"].get<std::vector<real_t>>();
    const std::vector<real_t> size
        = config["size"].get<std::vector<real_t>>();
    if (corner.size() != Dim) {
      fmt::print(stderr,
                 "Corner of cube initializer must have {} components instead "
                 "of {}\n",
                 Dim,
                 corner.size());
      exit(1);
    }
    if (size.size() != Dim) {
      fmt::print(stderr,
                 "Size of cube initializer must have {} components instead "
                 "of {}\n",
                 Dim,
                 size.size());
      exit(1);
    }
    if constexpr (Dim == 2) {
      return std::make_shared<Cube<Dim>>(
          std::array<real_t, 2>{corner[0], corner[1]},
	  std::array<real_t, 2>{size[0], size[1]}
	);
    } else {
      return std::make_shared<Cube<Dim>>(
          std::array<real_t, 3>{corner[0], corner[1], corner[2]},
          std::array<real_t, 3>{size[0], size[1], size[2]}
	);
    }
  } else {
    fmt::print(stderr, "Cube initializer is only defined for 2D or 3D\n");
    exit(1);
  }
}

}

#endif
