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
#ifndef UNIFORM_FACTORY_H_
#define UNIFORM_FACTORY_H_

#include "uniform.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <typename Result, typename RNG>
RandomVariable<Result> make_uniform(const nlohmann::json &config, RNG &rng) {
  Result min = 0;
  Result max = 1;
  if (config.contains("min")) {
    min = config["min"];
  }
  if (config.contains("max")) {
    max = config["max"];
  }
  return RandomVariable<Result>(
      std::make_shared<Uniform<Result, RNG>>(min, max, rng));
}

}

#endif
