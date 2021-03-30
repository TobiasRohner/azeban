#ifndef NORMAL_FACTORY_H_
#define NORMAL_FACTORY_H_

#include "normal.hpp"
#include <nlohmann/json.hpp>

namespace azeban {

template <typename Result, typename RNG>
RandomVariable<Result> make_normal(const nlohmann::json &config, RNG &rng) {
  Result mu = 0;
  Result sigma = 1;
  if (config.contains("mu")) {
    mu = config["mu"];
  }
  if (config.contains("sigma")) {
    sigma = config["sigma"];
  }
  return RandomVariable<Result>(
      std::make_shared<Normal<Result, RNG>>(mu, sigma, rng));
}

}

#endif
