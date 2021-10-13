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
#ifndef DOUBLE_SHEAR_LAYER_FACTORY_H_
#define DOUBLE_SHEAR_LAYER_FACTORY_H_

#include "double_shear_layer.hpp"
#include "init_3d_from_2d.hpp"
#include <azeban/logging.hpp>
#include <azeban/random/random_variable_factory.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>>
make_double_shear_layer(const nlohmann::json &config, RNG &rng) {
  if constexpr (Dim == 2 || Dim == 3) {
    AZEBAN_ERR_IF(
        !config.contains("rho"),
        "Double Shear Layer initialization is missing parameter \"rho\"\n");

    AZEBAN_ERR_IF(
        !config.contains("delta"),
        "Double Shear Layer initialization is missing parameter \"delta\"\n");

    RandomVariable<real_t> rho
        = make_random_variable<real_t>(config["rho"], rng);
    RandomVariable<real_t> delta
        = make_random_variable<real_t>(config["delta"], rng);
    if constexpr (Dim == 2) {
      return std::make_shared<DoubleShearLayer>(rho, delta);
    } else {
      AZEBAN_ERR_IF(
          !config.contains("dimension"),
          "Must specify constant \"dimension\" to generalize from 2D to 3D\n");

      const int dim = config["dimension"];
      auto init2d = std::make_shared<DoubleShearLayer>(rho, delta);
      return std::make_shared<Init3DFrom2D>(dim, init2d);
    }
  }

  AZEBAN_ERR("Double Shear Layer is only available for 2D or 3D simulations\n");
}

}

#endif
