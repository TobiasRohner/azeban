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
#ifndef EQUATION_MPI_FACTORY_H_
#define EQUATION_MPI_FACTORY_H_

#include <azeban/equations/equation.hpp>
#include <azeban/equations/incompressible_euler_mpi_factory.hpp>
#include <azeban/equations/spectral_viscosity_factory.hpp>
#include <azeban/grid.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <string>

namespace azeban {

template <int Dim>
std::shared_ptr<Equation<Dim>> make_equation_mpi(const nlohmann::json &config,
                                                 const Grid<Dim> &grid,
                                                 MPI_Comm comm,
                                                 bool has_tracer) {
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

  if (visc_type == "Smooth Cutoff") {
    SmoothCutoff1D visc = make_smooth_cutoff_1d(config["visc"], grid);

    if (equation_name == "Euler") {
      return make_incompressible_euler_mpi(grid, comm, visc, has_tracer);
    } else {
      fmt::print(stderr, "Unknown Equation");
      exit(1);
    }
  } else if (visc_type == "Step") {
    Step1D visc = make_step_1d(config["visc"], grid);

    if (equation_name == "Euler") {
      return make_incompressible_euler_mpi(grid, comm, visc, has_tracer);
    } else {
      fmt::print(stderr, "Unknown Equation");
      exit(1);
    }
  } else if (visc_type == "Quadratic") {
    Quadratic visc = make_quadratic(config["visc"], grid);

    if (equation_name == "Euler") {
      return make_incompressible_euler_mpi(grid, comm, visc, has_tracer);
    } else {
      fmt::print(stderr, "Unknown Equation");
      exit(1);
    }
  } else {
    fmt::print(stderr, "Unknown Spectral Viscosity type\n");
    exit(1);
  }
  // Make compiler happy
  return nullptr;
}

}

#endif
