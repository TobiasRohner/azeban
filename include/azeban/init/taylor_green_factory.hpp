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
#ifndef TAYLOR_GREEN_FACTORY_H_
#define TAYLOR_GREEN_FACTORY_H_

#include "taylor_green.hpp"
#include <azeban/random/delta.hpp>
#include <azeban/random/random_variable_factory.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>>
make_taylor_green(const nlohmann::json &config, RNG &rng) {
  if constexpr (Dim == 2 || Dim == 3) {
    if (config.contains("perturb")) {
      RandomVariable<real_t> delta
          = make_random_variable<real_t>(config["perturb"], rng);
      return std::make_shared<TaylorGreen<Dim>>(delta);
    } else {
      return std::make_shared<TaylorGreen<Dim>>();
    }
  } else {
    fmt::print(stderr,
               "Taylor Green  is only available for 2D or 3D simulations\n");
    exit(1);
  }
}

}

#endif
