#ifndef DELTA_FACTORY_H_
#define DELTA_FACTORY_H_

#include "delta.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <typename Result>
RandomVariable<Result> make_delta(const nlohmann::json &config) {
  if (config.is_number()) {
    return RandomVariable<Result>(
        std::make_shared<Delta<Result>>(config.get<Result>()));
  } else {
    if (!config.contains("value")) {
      fmt::print(
          stderr,
          "Delta Distribution configuration is missing parameter \"value\"\n");
      exit(1);
    }
    Result value = config["value"];
    return RandomVariable<Result>(std::make_shared<Delta<Result>>(value));
  }
}

}

#endif
