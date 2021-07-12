#ifndef CONST_PHYS_FACTORY_HPP_
#define CONST_PHYS_FACTORY_HPP_

#include "const_phys.hpp"
#include <azeban/random/random_variable_factory.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>


namespace azeban {

template<int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>>
make_const_phys(const nlohmann::json &config, RNG &rng) {
  if constexpr (Dim == 2) {
    if (!config.contains("u")) {
      fmt::print(stderr, "ConstPhys initialization is missing parameter \"u\"");
    }
    if (!config.contains("v")) {
      fmt::print(stderr, "ConstPhys initialization is missing parameter \"v\"");
    }
    RandomVariable<real_t> u = make_random_variable<real_t>(config["u"], rng);
    RandomVariable<real_t> v = make_random_variable<real_t>(config["v"], rng);
    return std::make_shared<ConstPhys<2>>(u, v);
  }
  else if constexpr (Dim == 3) {
    if (!config.contains("u")) {
      fmt::print(stderr, "ConstPhys initialization is missing parameter \"u\"");
    }
    if (!config.contains("v")) {
      fmt::print(stderr, "ConstPhys initialization is missing parameter \"v\"");
    }
    if (!config.contains("w")) {
      fmt::print(stderr, "ConstPhys initialization is missing parameter \"w\"");
    }
    RandomVariable<real_t> u = make_random_variable<real_t>(config["u"], rng);
    RandomVariable<real_t> v = make_random_variable<real_t>(config["v"], rng);
    RandomVariable<real_t> w = make_random_variable<real_t>(config["w"], rng);
    return std::make_shared<ConstPhys<3>>(u, v, w);
  }
  else {
    fmt::print(stderr, "ConstPhys is only implemented for 2D and 3D simulations");
    exit(1);
  }
}

}


#endif
