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
#ifndef RANDOM_VARIABLE_FACTORY_H_
#define RANDOM_VARIABLE_FACTORY_H_

#include "delta_factory.hpp"
#include "normal_factory.hpp"
#include "random_variable.hpp"
#include "uniform_factory.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <typename Result, typename RNG>
RandomVariable<Result> make_random_variable(const nlohmann::json &config,
                                            RNG &rng) {
  if (config.is_number()) {
    return RandomVariable<Result>(
        std::make_shared<Delta<Result>>(config.get<Result>()));
  } else {
    if (!config.contains("name")) {
      fmt::print(stderr,
                 "RandomVariable factory is missing parameter \"name\"\n");
      exit(1);
    }
    std::string name = config["name"];
    if (name == "Delta") {
      return make_delta<Result>(config);
    } else if (name == "Uniform") {
      return make_uniform<Result>(config, rng);
    } else if (name == "Normal") {
      return make_normal<Result>(config, rng);
    } else {
      fmt::print(stderr, "Unknown Random Variable name: {}\n", name);
      exit(1);
    }
  }
  // Make compiler happy
  return RandomVariable<Result>(nullptr);
}

}

#endif
