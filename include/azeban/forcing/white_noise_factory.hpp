#ifndef AZEBAN_FORCING_WHITE_NOISE_FACTORY_HPP_
#define AZEBAN_FORCING_WHITE_NOISE_FACTORY_HPP_

#include "white_noise.hpp"
#include <nlohmann/json.hpp>

namespace azeban {

template <typename RNG, int Dim>
WhiteNoise<Dim, RNG> make_white_noise(const nlohmann::json &config,
                                      const Grid<Dim> &grid,
                                      size_t seed) {
  if (!config.contains("sigma")) {
    fmt::print(stderr,
               "Must specify the standard deviation of the White Noise in key "
               "\"sigma\"\n");
    exit(1);
  }
  if (!config.contains("N")) {
    fmt::print(stderr, "Must specify the number of modes in key \"N\"\n");
    exit(1);
  }

  const real_t sigma = config["sigma"];
  const int N = config["N"];

  return WhiteNoise<Dim, RNG>(grid, sigma, N, seed);
}

}

#endif
