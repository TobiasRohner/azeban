#ifndef INCOMPRESSIBLE_EULER_MPI_FACTORY_H_
#define INCOMPRESSIBLE_EULER_MPI_FACTORY_H_

#include "incompressible_euler_mpi.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename SpectralViscosity>
std::shared_ptr<Equation<Dim>>
make_incompressible_euler_mpi(const Grid<Dim> &grid,
                              MPI_Comm comm,
                              const SpectralViscosity &visc,
                              bool has_tracer) {
  if constexpr (Dim == 2 || Dim == 3) {
    return std::make_shared<IncompressibleEuler_MPI<Dim, SpectralViscosity>>(
        grid, comm, visc, has_tracer);
  } else {
    ZISA_UNUSED(grid);
    ZISA_UNUSED(comm);
    ZISA_UNUSED(visc);
    ZISA_UNUSED(has_tracer);
    fmt::print(stderr, "Euler is only implemented for 2D or 3D\n");
    exit(1);
  }
}

}

#endif
