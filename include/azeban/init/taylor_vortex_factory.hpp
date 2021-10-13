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
#ifndef TAYLOR_VORTEX_FACTORY_H_
#define TAYLOR_VORTEX_FACTORY_H_

#include "init_3d_from_2d.hpp"
#include "taylor_vortex.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>>
make_taylor_vortex(const nlohmann::json &config) {
  if constexpr (Dim == 2) {
    return std::make_shared<TaylorVortex>();
  } else if constexpr (Dim == 3) {
    if (!config.contains("dimension")) {
      fmt::print(stderr,
                 "Must specify constant \"dimension\" to generalize from 2D "
                 "to 3D\n");
      exit(1);
    }
    const int dim = config["dimension"];
    const auto init2d = std::make_shared<TaylorVortex>();
    return std::make_shared<Init3DFrom2D>(dim, init2d);
  } else {
    fmt::print(stderr,
               "Taylor Vortex is only available for 2D or 3D simulations\n");
    exit(1);
  }
}

}

#endif
