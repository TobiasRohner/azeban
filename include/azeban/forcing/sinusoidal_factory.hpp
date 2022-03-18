#ifndef AZEBAN_FORCING_SINUSOIDAL_FACTORY_HPP_
#define AZEBAN_FORCING_SINUSOIDAL_FACTORY_HPP_

#include <azeban/forcing/sinusoidal.hpp>

namespace azeban {

template <int Dim>
Sinusoidal make_sinusoidal(const nlohmann::json &config,
                           const Grid<Dim> &grid) {
  if (!config.contains("amplitude")) {
    fmt::print(stderr, "Sinusoidal Forcing needs an amplitude in the config\n");
    exit(1);
  }
  const real_t amplitude = config["amplitude"];
  return Sinusoidal(grid.N_phys, amplitude);
}

}

#endif
