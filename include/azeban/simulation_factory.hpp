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
#ifndef SIMULATION_FACTORY_H_
#define SIMULATION_FACTORY_H_

#include <azeban/equations/equation_factory.hpp>
#include <azeban/evolution/cfl.hpp>
#include <azeban/evolution/time_integrator_factory.hpp>
#include <azeban/grid_factory.hpp>
#include <azeban/simulation.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <string>
#include <zisa/config.hpp>

namespace azeban {

template <int Dim>
Simulation<Dim> make_simulation(const nlohmann::json &config) {
  zisa::device_type device;
  if (!config.contains("device")) {
    fmt::print(stderr, "Device not specified. Defaulting to CPU");
    device = zisa::device_type::cpu;
  } else {
    const std::string device_name = config["device"];
    if (device_name == "cpu") {
      device = zisa::device_type::cpu;
    } else if (device_name == "cuda") {
      device = zisa::device_type::cuda;
    } else {
      fmt::print(stderr, "Unknown device type: {}\n", device_name);
      exit(1);
    }
  }

  if (!config.contains("grid")) {
    fmt::print(stderr, "Config must contain key \"grid\"\n");
    exit(1);
  }
  auto grid = make_grid<Dim>(config["grid"], device);

  if (!config.contains("equation")) {
    fmt::print(stderr, "Config must contain key \"equation\"\n");
    exit(1);
  }
  const bool has_tracer
      = config.contains("init") && config["init"].contains("tracer");
  auto equation
      = make_equation<Dim>(config["equation"], grid, has_tracer, device);

  if (!config.contains("timestepper")) {
    fmt::print("Config is missing timestepper specifications\n");
    exit(1);
  }
  auto timestepper
      = make_timestepper(config["timestepper"], grid, equation, device);

  if (!config["timestepper"].contains("C")) {
    fmt::print(stderr, "Timestepper config is missing CFL constant \"C\"\n");
    exit(1);
  }
  const real_t C = config["timestepper"]["C"];
  auto cfl = CFL<Dim>(grid, C);

  return Simulation(
      grid.shape_fourier(equation->n_vars()), cfl, timestepper, device);
}

}

#endif
