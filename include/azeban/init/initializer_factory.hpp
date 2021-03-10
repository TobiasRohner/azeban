#ifndef INITIALIZER_FACTORY_H_
#define INITIALIZER_FACTORY_H_

#include "discontinuous_vortex_patch.hpp"
#include "double_shear_layer.hpp"
#include "init_3d_from_2d.hpp"
#include "initializer.hpp"
#include "shock.hpp"
#include "sine_1d.hpp"
#include "taylor_vortex.hpp"
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
    if constexpr (Dim == 1) {
      return std::make_shared<Sine1D>();
    } else {
      fmt::print(stderr, "Sine 1D is only available for 1D simulations\n");
      exit(1);
    }
  } else if (name == "Shock") {
    if constexpr (Dim == 1) {
      if (!config["init"].contains("x0")) {
        fmt::print(stderr,
                   "Shock initialization is missing parameter \"x0\"\n");
        exit(1);
      }
      if (!config["init"].contains("x1")) {
        fmt::print(stderr,
                   "Shock initialization is missing parameter \"x1\"\n");
        exit(1);
      }
      const real_t x0 = config["init"]["x0"];
      const real_t x1 = config["init"]["x1"];
      return std::make_shared<Shock>(x0, x1);
    } else {
      fmt::print(stderr, "Sine 1D is only available for 1D simulations\n");
      exit(1);
    }
  } else if (name == "Double Shear Layer") {
    if constexpr (Dim == 2 || Dim == 3) {
      if (!config["init"].contains("rho")) {
        fmt::print(
            stderr,
            "Double Shear Layer initialization is missing parameter \"rho\"\n");
        exit(1);
      }
      if (!config["init"].contains("delta")) {
        fmt::print(stderr,
                   "Double Shear Layer initialization is missing parameter "
                   "\"delta\"\n");
        exit(1);
      }
      const real_t rho = config["init"]["rho"];
      const real_t delta = config["init"]["delta"];
      if constexpr (Dim == 2) {
        return std::make_shared<DoubleShearLayer>(rho, delta);
      } else {
        if (!config["init"].contains("dimension")) {
          fmt::print(stderr,
                     "Must specify constant \"dimension\" to generalize from "
                     "2D to 3D\n");
          exit(1);
        }
        const int dim = config["init"]["dimension"];
        const auto init2d = std::make_shared<DoubleShearLayer>(rho, delta);
        return std::make_shared<Init3DFrom2D>(dim, init2d);
      }
    } else {
      fmt::print(
          stderr,
          "Double Shear Layer is only available for 2D or 3D simulations\n");
      exit(1);
    }
  } else if (name == "Taylor Vortex") {
    if constexpr (Dim == 2) {
      return std::make_shared<TaylorVortex>();
    } else if constexpr (Dim == 3) {
      if (!config["init"].contains("dimension")) {
        fmt::print(stderr,
                   "Must specify constant \"dimension\" to generalize from 2D "
                   "to 3D\n");
        exit(1);
      }
      const int dim = config["init"]["dimension"];
      const auto init2d = std::make_shared<TaylorVortex>();
      return std::make_shared<Init3DFrom2D>(dim, init2d);
    } else {
      fmt::print(stderr,
                 "Taylor Vortex is only available for 2D or 3D simulations\n");
      exit(1);
    }
  } else if (name == "Discontinuous Vortex Patch") {
    if constexpr (Dim == 2) {
      return std::make_shared<DiscontinuousVortexPatch>();
    } else if constexpr (Dim == 3) {
      if (!config["init"].contains("dimension")) {
        fmt::print(stderr,
                   "Must specify constant \"dimension\" to generalize from 2D "
                   "to 3D\n");
        exit(1);
      }
      const int dim = config["init"]["dimension"];
      const auto init2d = std::make_shared<DiscontinuousVortexPatch>();
      return std::make_shared<Init3DFrom2D>(dim, init2d);
    } else {
      fmt::print(stderr,
                 "Discontinuous Vortex Patch is only available for 2D or 3D "
                 "simulations\n");
      exit(1);
    }
  } else {
    fmt::print(stderr, "Unknown Initializer: \"{}\"\n", name);
    exit(1);
  }
}

}

#endif
