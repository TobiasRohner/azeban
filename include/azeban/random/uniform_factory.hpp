#ifndef UNIFORM_FACTORY_H_
#define UNIFORM_FACTORY_H_

#include "uniform.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <typename Result, typename RNG>
RandomVariable<Result> make_uniform(const nlohmann::json &config, RNG &rng) {
  if (!config.contains("min")) {
    fmt::print(
        stderr,
        "Uniform Distribution configuration is missing parameter \"min\"\n");
    exit(1);
  }
  if (!config.contains("max")) {
    fmt::print(
        stderr,
        "Uniform Distribution configuration is missing parameter \"max\"\n");
    exit(1);
  }
  Result min = config["min"];
  Result max = config["max"];
  return RandomVariable<Result>(
      std::make_shared<Uniform<Result, RNG>>(min, max, rng));
}

}

#endif
