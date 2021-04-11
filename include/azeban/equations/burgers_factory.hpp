#ifndef BURGERS_FACTORY_H_
#define BURGERS_FACTORY_H_

#include "burgers.hpp"

namespace azeban {

template <int Dim, typename SpectralViscosity>
std::shared_ptr<Equation<Dim>> make_burgers(const Grid<Dim> &grid,
                                            const SpectralViscosity &visc,
                                            zisa::device_type device) {
  if constexpr (Dim == 1) {
    return std::make_shared<Burgers<SpectralViscosity>>(grid, visc, device);
  } else {
    ZISA_UNUSED(grid);
    ZISA_UNUSED(visc);
    ZISA_UNUSED(device);
    fmt::print(stderr, "Burgers is only implemented for 1D\n");
    exit(1);
  }
}

}

#endif
