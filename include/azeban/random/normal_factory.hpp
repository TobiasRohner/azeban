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
#ifndef NORMAL_FACTORY_H_
#define NORMAL_FACTORY_H_

#include "normal.hpp"
#include <nlohmann/json.hpp>

namespace azeban {

template <typename Result, typename RNG>
RandomVariable<Result> make_normal(const nlohmann::json &config, RNG &rng) {
  Result mu = 0;
  Result sigma = 1;
  if (config.contains("mu")) {
    mu = config["mu"];
  }
  if (config.contains("sigma")) {
    sigma = config["sigma"];
  }
  return RandomVariable<Result>(
      std::make_shared<Normal<Result, RNG>>(mu, sigma, rng));
}

}

#endif
