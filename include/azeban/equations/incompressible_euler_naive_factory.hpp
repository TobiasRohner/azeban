#ifndef INCOMPRESSIBLE_EULER_NAIVE_FACTORY_H_
#define INCOMPRESSIBLE_EULER_NAIVE_FACTORY_H_

#include "incompressible_euler_naive.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename SpectralViscosity>
std::shared_ptr<Equation<Dim>>
make_incompressible_euler_naive(const Grid<Dim> &grid,
                                const SpectralViscosity &visc,
                                zisa::device_type device) {
  if constexpr (Dim == 2 || Dim == 3) {
    return std::make_shared<IncompressibleEulerNaive<Dim, SpectralViscosity>>(
        grid, visc, device);
  } else {
    ZISA_UNUSED(grid);
    ZISA_UNUSED(visc);
    ZISA_UNUSED(device);
    fmt::print(stderr, "Euler is only implemented for 2D or 3D\n");
    exit(1);
  }
}

}

#endif
