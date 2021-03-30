#ifndef BROWNIAN_MOTION_FACTORY_H_
#define BROWNIAN_MOTION_FACTORY_H_

#include "brownian_motion.hpp"
#include <azeban/random/random_variable_factory.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>>
make_brownian_motion(const nlohmann::json &config, RNG &rng) {
  RandomVariable<real_t> H(std::make_shared<Delta<real_t>>(0.5));
  if (config.contains("hurst")) {
    H = make_random_variable<real_t>(config["hurst"], rng);
  }
  return std::make_shared<BrownianMotion<Dim>>(H, rng);
}

}

#endif
