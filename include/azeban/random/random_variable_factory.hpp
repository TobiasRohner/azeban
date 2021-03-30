#ifndef RANDOM_VARIABLE_FACTORY_H_
#define RANDOM_VARIABLE_FACTORY_H_

#include "delta_factory.hpp"
#include "normal_factory.hpp"
#include "random_variable.hpp"
#include "uniform_factory.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <typename Result, typename RNG>
RandomVariable<Result> make_random_variable(const nlohmann::json &config,
                                            RNG &rng) {
  if (config.is_number()) {
    return RandomVariable<Result>(
        std::make_shared<Delta<Result>>(config.get<Result>()));
  } else {
    if (!config.contains("name")) {
      fmt::print(stderr,
                 "RandomVariable factory is missing parameter \"name\"\n");
      exit(1);
    }
    std::string name = config["name"];
    if (name == "Delta") {
      return make_delta<Result>(config);
    } else if (name == "Uniform") {
      return make_uniform<Result>(config, rng);
    } else if (name == "Normal") {
      return make_normal<Result>(config, rng);
    } else {
      fmt::print(stderr, "Unknown Random Variable name: {}\n", name);
      exit(1);
    }
  }
  // Make compiler happy
  return RandomVariable<Result>(nullptr);
}

}

#endif
