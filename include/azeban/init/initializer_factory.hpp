#ifndef INITIALIZER_FACTORY_H_
#define INITIALIZER_FACTORY_H_

#include "discontinuous_vortex_patch_factory.hpp"
#include "double_shear_layer_factory.hpp"
#include "init_3d_from_2d.hpp"
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

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>>
make_initializer_u(const nlohmann::json &config) {
  if (!config.contains("name")) {
    fmt::print(stderr, "Config does not contain initializer name\n");
    exit(1);
  }

  std::string name = config["name"];
  if (name == "Sine 1D") {
    return make_sine_1d<Dim>();
  } else if (name == "Shock") {
    return make_shock<Dim>(config);
  } else if (name == "Double Shear Layer") {
    return make_double_shear_layer<Dim>(config);
  } else if (name == "Taylor Vortex") {
    return make_taylor_vortex<Dim>(config);
  } else if (name == "Discontinuous Vortex Patch") {
    return make_discontinuous_vortex_patch<Dim>(config);
  } else if (name == "Taylor Green") {
    return make_taylor_green<Dim>();
  } else if (name == "Shear Tube") {
    return make_shear_tube<Dim>(config);
  } else {
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
  } else {
    fmt::print(stderr, "Unknown Initializer: \"{}\"\n", name);
    exit(1);
  }
}

template <int Dim>
std::shared_ptr<Initializer<Dim>>
make_initializer(const nlohmann::json &config) {
  if (!config.contains("init")) {
    fmt::print(stderr, "Config does not contain initialization information\n");
    exit(1);
  }

  auto init_u = make_initializer_u<Dim>(config["init"]);
  if (config["init"].contains("tracer")) {
    auto init_rho = make_initializer_rho<Dim>(config["init"]["tracer"]);
    return std::make_shared<VelocityAndTracer<Dim>>(init_u, init_rho);
  } else {
    return init_u;
  }
}

}

#endif
