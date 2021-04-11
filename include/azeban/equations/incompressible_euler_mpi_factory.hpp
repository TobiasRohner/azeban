#ifndef INCOMPRESSIBLE_EULER_MPI_FACTORY_H_
#define INCOMPRESSIBLE_EULER_MPI_FACTORY_H_

#include "incompressible_euler_mpi.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename SpectralViscosity>
std::shared_ptr<Equation<Dim>>
make_incompressible_euler_mpi(const Grid_MPI<Dim> &grid,
                              const SpectralViscosity &visc,
                              bool has_tracer) {
  if constexpr (Dim == 2 || Dim == 3) {
    return std::make_shared<IncompressibleEuler_MPI<Dim, SpectralViscosity>>(grid, visc, has_tracer);
  } else {
    fmt::print(stderr, "Euler is only implemented for 2D or 3D\n");
    exit(1);
  }
}

}

#endif
