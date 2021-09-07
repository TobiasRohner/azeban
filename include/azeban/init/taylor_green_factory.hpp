#ifndef TAYLOR_GREEN_FACTORY_H_
#define TAYLOR_GREEN_FACTORY_H_

#include "taylor_green.hpp"
#include <azeban/random/random_variable_factory.hpp>
#include <azeban/random/delta.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>> make_taylor_green(const nlohmann::json &config, RNG &rng) {
  if constexpr (Dim == 2 || Dim == 3) {
    if (config.contains("perturb")) {
      RandomVariable<real_t> delta = make_random_variable<real_t>(config["perturb"], rng);
      return std::make_shared<TaylorGreen<Dim>>(delta);
    }
    else {
      return std::make_shared<TaylorGreen<Dim>>();
    }
  } else {
    fmt::print(stderr,
               "Taylor Green  is only available for 2D or 3D simulations\n");
    exit(1);
  }
}

}

#endif
