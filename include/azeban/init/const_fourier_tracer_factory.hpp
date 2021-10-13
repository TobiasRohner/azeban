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
#ifndef CONST_FOURIER_TRACER_FACTORY_HPP_
#define CONST_FOURIER_TRACER_FACTORY_HPP_

#include "const_fourier_tracer.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>>
make_const_fourier_tracer(const nlohmann::json &config) {
  if (!config.contains("rho")) {
    fmt::print(stderr,
               "ConstFourier initialization is missing parameter \"rho\"");
    exit(1);
  }
  real_t rho = config["rho"];
  return std::make_shared<ConstFourierTracer<Dim>>(rho);
}

}

#endif
