#ifndef SHOCK_FACTORY_H_
#define SHOCK_FACTORY_H_

#include "shock.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>> make_shock(const nlohmann::json &config) {
  if constexpr (Dim == 1) {
    if (!config.contains("x0")) {
      fmt::print(stderr, "Shock initialization is missing parameter \"x0\"\n");
      exit(1);
    }
    if (!config.contains("x1")) {
      fmt::print(stderr, "Shock initialization is missing parameter \"x1\"\n");
      exit(1);
    }
    const real_t x0 = config["x0"];
    const real_t x1 = config["x1"];
    return std::make_shared<Shock>(x0, x1);
  } else {
    fmt::print(stderr, "Shock is only available for 1D simulations\n");
    exit(1);
  }
}

}

#endif
