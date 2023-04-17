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
#ifndef INITIALIZER_FACTORY_H_
#define INITIALIZER_FACTORY_H_

#include "brownian_motion_factory.hpp"
#include "const_fourier_tracer_factory.hpp"
#include "const_phys_factory.hpp"
#include "discontinuous_double_shear_layer_factory.hpp"
#include "discontinuous_shear_tube_factory.hpp"
#include "discontinuous_vortex_patch_factory.hpp"
#include "double_shear_layer_factory.hpp"
#include "init_3d_from_2d.hpp"
#include "init_from_file_factory.hpp"
#include "initializer.hpp"
#include "shear_tube_factory.hpp"
#include "shock_factory.hpp"
#include "sine_1d_factory.hpp"
#include "sphere_factory.hpp"
#include "taylor_green_factory.hpp"
#include "taylor_vortex_factory.hpp"
#include "velocity_and_tracer.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <string>
#include <zisa/config.hpp>
#if AZEBAN_HAS_PYTHON
#include "python_factory.hpp"
#endif

namespace azeban {

template <int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>>
make_initializer_u(const nlohmann::json &config, RNG &rng) {
  if (!config.contains("name")) {
    fmt::print(stderr, "Config does not contain initializer name\n");
    exit(1);
  }

  std::string name = config["name"];
  if (name == "Sine 1D") {
    return make_sine_1d<Dim>();
  } else if (name == "Shock") {
    return make_shock<Dim>(config, rng);
  } else if (name == "Double Shear Layer") {
    return make_double_shear_layer<Dim>(config, rng);
  } else if (name == "Discontinuous Double Shear Layer") {
    return make_discontinuous_double_shear_layer<Dim>(config, rng);
  } else if (name == "Taylor Vortex") {
    return make_taylor_vortex<Dim>(config);
  } else if (name == "Discontinuous Vortex Patch") {
    return make_discontinuous_vortex_patch<Dim>(config);
  } else if (name == "Taylor Green") {
    return make_taylor_green<Dim>(config, rng);
  } else if (name == "Shear Tube") {
    return make_shear_tube<Dim>(config, rng);
  } else if (name == "Discontinuous Shear Tube") {
    return make_discontinuous_shear_tube<Dim>(config, rng);
  } else if (name == "Brownian Motion") {
    return make_brownian_motion<Dim>(config, rng);
  } else if (name == "Const Phys") {
    return make_const_phys<Dim>(config, rng);
  } else if (name == "Init From File") {
    return make_init_from_file<Dim>(config);
  } 
# if AZEBAN_HAS_PYTHON
  else if (name == "Python") {
    return make_python<Dim>(config, rng);
  }
#endif
  else {
    fmt::print(stderr, "Unknown Initializer: \"{}\"\n", name);
    exit(1);
  }
}

template <int Dim>
std::shared_ptr<Initializer<Dim>>
make_initializer_rho(const nlohmann::json &config) {
  if (!config.contains("name")) {
    fmt::print(stderr, "Config does not contain initializer name\n");
    exit(1);
  }

  std::string name = config["name"];
  if (name == "Sphere") {
    return make_sphere<Dim>(config);
  } else if (name == "Const Fourier") {
    return make_const_fourier_tracer<Dim>(config);
  } else {
    fmt::print(stderr, "Unknown Initializer: \"{}\"\n", name);
    exit(1);
  }
}

template <int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>> make_initializer(const nlohmann::json &config,
                                                   RNG &rng) {
  if (!config.contains("init")) {
    fmt::print(stderr, "Config does not contain initialization information\n");
    exit(1);
  }

  auto init_u = make_initializer_u<Dim>(config["init"], rng);
  if (config["init"].contains("tracer")) {
    auto init_rho = make_initializer_rho<Dim>(config["init"]["tracer"]);
    return std::make_shared<VelocityAndTracer<Dim>>(init_u, init_rho);
  } else {
    return init_u;
  }
}

}

#endif
