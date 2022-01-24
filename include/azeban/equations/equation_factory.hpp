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
#ifndef EQUATION_FACTORY_H_
#define EQUATION_FACTORY_H_

#include <azeban/equations/burgers_factory.hpp>
#include <azeban/equations/equation.hpp>
#include <azeban/equations/incompressible_euler_factory.hpp>
#include <azeban/equations/incompressible_euler_naive_factory.hpp>
#include <azeban/equations/spectral_viscosity_factory.hpp>
#include <azeban/grid.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <string>

namespace azeban {

template <int Dim>
std::shared_ptr<Equation<Dim>> make_equation(const nlohmann::json &config,
                                             const Grid<Dim> &grid,
                                             bool has_tracer,
                                             zisa::device_type device) {
  if (!config.contains("name")) {
    fmt::print(stderr, "Equation config must contain key \"name\"\n");
    exit(1);
  }
  if (!config.contains("visc")) {
    fmt::print(stderr, "Equation config must contain key \"visc\"\n");
    exit(1);
  }
  if (!config["visc"].contains("type")) {
    fmt::print(stderr,
               "Must specify the type of Spectral Viscosity in key \"type\"\n");
    exit(1);
  }

  const std::string equation_name = config["name"];
  const std::string visc_type = config["visc"]["type"];

  auto make_equation
      = [&equation_name, &grid, &has_tracer, &device](auto visc) {
          if (equation_name == "Burgers") {
            return make_burgers(grid, visc, device);
          } else if (equation_name == "Euler") {
            return make_incompressible_euler(grid, visc, has_tracer, device);
          } else if (equation_name == "Euler Naive") {
            return make_incompressible_euler_naive(grid, visc, device);
          }

          AZEBAN_ERR("Unkown Equation");
        };

  if (visc_type == "Smooth Cutoff") {
    SmoothCutoff1D visc = make_smooth_cutoff_1d(config["visc"], grid);
    return make_equation(visc);

  } else if (visc_type == "Step") {
    Step1D visc = make_step_1d(config["visc"], grid);
    return make_equation(visc);

  } else if (visc_type == "Quadratic") {
    Quadratic visc = make_quadratic(config["visc"], grid);
    return make_equation(visc);

  } else if (visc_type == "None") {
    NoViscosity visc = make_no_viscosity(config["visc"], grid);
    return make_equation(visc);
  }

  AZEBAN_ERR("Unknown Spectral Viscosity type.\n");
}

}
#endif
