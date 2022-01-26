#ifndef AZEBAN_FORCING_WHITE_NOISE_FACTORY_HPP_
#define AZEBAN_FORCING_WHITE_NOISE_FACTORY_HPP_

#include "white_noise.hpp"
#include <nlohmann/json.hpp>

namespace azeban {

template <typename RNG, int Dim>
WhiteNoise<RNG> make_white_noise(const nlohmann::json &config,
                                 const Grid<Dim> &grid) {
  if (!config.contains("sigma")) {
    fmt::print(stderr,
               "Must specify the standard deviation of the White Noise in key "
               "\"sigma\"\n");
    exit(1);
  }

  const real_t sigma = config["sigma"];
  unsigned long long seed = 0;
  if (config.contains("seed")) {
    seed = config["seed"];
  }

  return WhiteNoise<RNG>(grid, sigma, seed);
}

}

#endif
