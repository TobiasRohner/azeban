#ifndef INCOMPRESSIBLE_EULER_FACTORY_H_
#define INCOMPRESSIBLE_EULER_FACTORY_H_

#include "incompressible_euler.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename SpectralViscosity>
std::shared_ptr<Equation<Dim>>
make_incompressible_euler(const Grid<Dim> &grid,
                          const SpectralViscosity &visc,
                          bool has_tracer,
                          zisa::device_type device) {
  if constexpr (Dim == 2 || Dim == 3) {
    return std::make_shared<IncompressibleEuler<Dim, SpectralViscosity>>(
        grid, visc, device, has_tracer);
  } else {
    fmt::print(stderr, "Euler is only implemented for 2D or 3D\n");
    exit(1);
  }
}

}

#endif
