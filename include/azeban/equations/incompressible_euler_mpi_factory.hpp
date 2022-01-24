/*
 * This file is part of azeban (https://github.com/TobiasRohner/azeban).
 * Copyright (c) 2021 Tobias Rohner.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
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
