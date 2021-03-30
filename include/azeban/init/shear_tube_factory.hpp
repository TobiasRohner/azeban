#ifndef SHEAR_TUBE_FACTORY_H_
#define SHEAR_TUBE_FACTORY_H_

#include "shear_tube.hpp"
#include <azeban/random/random_variable_factory.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>> make_shear_tube(const nlohmann::json &config,
                                                  RNG &rng) {
  if constexpr (Dim == 3) {
    if (!config.contains("rho")) {
      fmt::print(stderr,
                 "Shear Tube initialization is missing parameter \"rho\"\n");
      exit(1);
    }
    if (!config.contains("delta")) {
      fmt::print(stderr,
                 "Shear Tube initialization is missing parameter "
                 "\"delta\"\n");
      exit(1);
    }
    RandomVariable<real_t> rho
        = make_random_variable<real_t>(config["rho"], rng);
    RandomVariable<real_t> delta
        = make_random_variable<real_t>(config["delta"], rng);
    return std::make_shared<ShearTube>(rho, delta);
  } else {
    fmt::print(stderr, "Shear Tube is only available for 3D simulations\n");
    exit(1);
  }
}

}

#endif
