#ifndef BURGERS_FACTORY_H_
#define BURGERS_FACTORY_H_

#include "burgers.hpp"

namespace azeban {

template <int Dim, typename SpectralViscosity>
std::shared_ptr<Equation<Dim>> make_burgers(const nlohmann::json &config,
                                            const Grid<Dim> &grid,
                                            const SpectralViscosity &visc,
                                            zisa::device_type device) {
  if constexpr (Dim == 1) {
    return std::make_shared<Burgers<SpectralViscosity>>(grid, visc, device);
  } else {
    fmt::print(stderr, "Burgers is only implemented for 1D\n");
    exit(1);
  }
}

}

#endif
