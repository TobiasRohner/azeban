#ifndef AZEBAN_FORCING_WHITE_NOISE_HIGH_FREQ_FACTORY_HPP_
#define AZEBAN_FORCING_WHITE_NOISE_HIGH_FREQ_FACTORY_HPP_

#include "white_noise_high_freq.hpp"
#include <nlohmann/json.hpp>

namespace azeban {

template <typename RNG, int Dim>
WhiteNoiseHighFreq<Dim, RNG>
make_white_noise_high_freq(const nlohmann::json &config,
                           const Grid<Dim> &grid,
                           real_t eps,
                           size_t seed) {
  if (!config.contains("k_min")) {
    fmt::print(stderr, "Must specify the minimal forced mode in \"k_min\"\n");
    exit(1);
  }
  if (!config.contains("k_max")) {
    fmt::print(stderr, "Must specify the maximal forced mode in \"k_max\"\n");
    exit(1);
  }

  const int k_min = config["k_min"];
  const int k_max = config["k_max"];
  real_t b = 1;
  if (config.contains("b")) {
    b = config["b"];
  }

  return WhiteNoiseHighFreq<Dim, RNG>(grid, b, k_min, k_max, eps, seed);
}

}

#endif
