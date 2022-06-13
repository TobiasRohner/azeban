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
#ifndef SPECTRAL_VISCOSITY_FACTORY_H_
#define SPECTRAL_VISCOSITY_FACTORY_H_

#include "spectral_viscosity.hpp"
#include <azeban/grid.hpp>
#include <azeban/logging.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
SmoothCutoff1D make_smooth_cutoff_1d(const nlohmann::json &config,
                                     const Grid<Dim> &grid) {
  real_t s = 1;
  if (config.contains("s")) {
    s = config["s"];
  }
  real_t theta = (2 * s - 1) / (2 * s);
  if (config.contains("theta")) {
    theta = config["theta"];
  }
  AZEBAN_ERR_IF(theta > (2 * s - 1) / (2 * s),
                "theta parameter of spectral viscosity is too big\n");
  real_t m0 = 1;
  if (config.contains("m0")) {
    m0 = config["m0"];
  }
  AZEBAN_ERR_IF(
      !config.contains("eps"),
      "Smooth Cutoff Viscosity expects \"eps\" parameter in config\n");
  const real_t eps = config["eps"];

  const real_t epsN
      = eps / zisa::pow(static_cast<real_t>(grid.N_phys), 2 * s - 1);
  const real_t mN = m0 * zisa::pow(static_cast<real_t>(grid.N_phys), theta);

  return SmoothCutoff1D(epsN, s, mN);
}

template <int Dim>
Step1D make_step_1d(const nlohmann::json &config, const Grid<Dim> &grid) {
  real_t s = 1;
  if (config.contains("s")) {
    s = config["s"];
  }
  real_t theta = (2 * s - 1) / (2 * s);
  if (config.contains("theta")) {
    theta = config["theta"];
  }
  AZEBAN_ERR_IF(theta > (2 * s - 1) / (2 * s),
                "theta parameter of spectral viscosity is too big\n");
  real_t m0 = 1;
  if (config.contains("m0")) {
    m0 = config["m0"];
  }
  AZEBAN_ERR_IF(
      !config.contains("eps"),
      "Smooth Cutoff Viscosity expects \"eps\" parameter in config\n");
  const real_t eps = config["eps"];

  const real_t epsN
      = eps / zisa::pow(static_cast<real_t>(grid.N_phys), 2 * s - 1);
  const real_t mN = m0 * zisa::pow(static_cast<real_t>(grid.N_phys), theta);

  return Step1D(epsN, s, mN);
}

template <int Dim>
Quadratic make_quadratic(const nlohmann::json &config, const Grid<Dim> &grid) {
  AZEBAN_ERR_IF(!config.contains("eps"),
                "Quadratic Viscosity expects \"eps\" parameter in config\n");

  const real_t eps = config["eps"];
  return Quadratic(eps / grid.N_phys, grid.N_phys);
}

template <int Dim>
NoViscosity make_no_viscosity(const nlohmann::json &config,
                              const Grid<Dim> & /* grid */) {

  AZEBAN_ERR_IF(config["type"] != "None",
                "Config file did not request \"type = None\".\n");

  return NoViscosity();
}

}

#endif
