#ifndef DOUBLE_SHEAR_LAYER_FACTORY_H_
#define DOUBLE_SHEAR_LAYER_FACTORY_H_

#include "double_shear_layer.hpp"
#include "init_3d_from_2d.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>>
make_double_shear_layer(const nlohmann::json &config) {
  if constexpr (Dim == 2 || Dim == 3) {
    if (!config.contains("rho")) {
      fmt::print(
          stderr,
          "Double Shear Layer initialization is missing parameter \"rho\"\n");
      exit(1);
    }
    if (!config.contains("delta")) {
      fmt::print(stderr,
                 "Double Shear Layer initialization is missing parameter "
                 "\"delta\"\n");
      exit(1);
    }
    const real_t rho = config["rho"];
    const real_t delta = config["delta"];
    if constexpr (Dim == 2) {
      return std::make_shared<DoubleShearLayer>(rho, delta);
    } else {
      if (!config.contains("dimension")) {
        fmt::print(stderr,
                   "Must specify constant \"dimension\" to generalize from "
                   "2D to 3D\n");
        exit(1);
      }
      const int dim = config["dimension"];
      const auto init2d = std::make_shared<DoubleShearLayer>(rho, delta);
      return std::make_shared<Init3DFrom2D>(dim, init2d);
    }
  } else {
    fmt::print(
        stderr,
        "Double Shear Layer is only available for 2D or 3D simulations\n");
    exit(1);
  }
}

}

#endif
