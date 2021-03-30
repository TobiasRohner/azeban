#ifndef SHOCK_FACTORY_H_
#define SHOCK_FACTORY_H_

#include "shock.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>> make_shock(const nlohmann::json &config,
                                             RNG &rng) {
  if constexpr (Dim == 1) {
    if (!config.contains("x0")) {
      fmt::print(stderr, "Shock initialization is missing parameter \"x0\"\n");
      exit(1);
    }
    if (!config.contains("x1")) {
      fmt::print(stderr, "Shock initialization is missing parameter \"x1\"\n");
      exit(1);
    }
    RandomVariable<real_t> x0 = make_random_variable<real_t>(config["x0"], rng);
    RandomVariable<real_t> x1 = make_random_variable<real_t>(config["x1"], rng);
    return std::make_shared<Shock>(x0, x1);
  } else {
    fmt::print(stderr, "Shock is only available for 1D simulations\n");
    exit(1);
  }
}

}

#endif
