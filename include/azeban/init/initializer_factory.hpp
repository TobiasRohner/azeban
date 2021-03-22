#ifndef INITIALIZER_FACTORY_H_
#define INITIALIZER_FACTORY_H_

#include "discontinuous_vortex_patch_factory.hpp"
#include "double_shear_layer_factory.hpp"
#include "init_3d_from_2d.hpp"
#include "initializer.hpp"
#include "shock_factory.hpp"
#include "sine_1d_factory.hpp"
#include "taylor_green_factory.hpp"
#include "taylor_vortex_factory.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <string>
#include <zisa/config.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>>
make_initializer(const nlohmann::json &config) {
  if (!config.contains("init")) {
    fmt::print(stderr, "Config does not contain initialization information\n");
    exit(1);
  }
  if (!config["init"].contains("name")) {
    fmt::print(stderr, "Config does not contain initializer name\n");
    exit(1);
  }

  std::string name = config["init"]["name"];
  if (name == "Sine 1D") {
    return make_sine_1d<Dim>(config["init"]);
  } else if (name == "Shock") {
    return make_shock<Dim>(config["init"]);
  } else if (name == "Double Shear Layer") {
    return make_double_shear_layer<Dim>(config["init"]);
  } else if (name == "Taylor Vortex") {
    return make_taylor_vortex<Dim>(config["init"]);
  } else if (name == "Discontinuous Vortex Patch") {
    return make_discontinuous_vortex_patch<Dim>(config["init"]);
  } else if (name == "Taylor Green") {
    return make_taylor_green<Dim>(config["init"]);
  } else {
    fmt::print(stderr, "Unknown Initializer: \"{}\"\n", name);
    exit(1);
  }
}

}

#endif
