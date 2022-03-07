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
#ifndef GRID_H_
#define GRID_H_

#include <azeban/config.hpp>
#include <zisa/config.hpp>
#include <zisa/memory/array.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/communicator.hpp>
#endif

namespace azeban {

template <int Dim>
struct Grid {
  static constexpr int dim_v = Dim;

  zisa::int_t N_phys;
  zisa::int_t N_phys_pad;
  zisa::int_t N_fourier;
  zisa::int_t N_fourier_pad;

  Grid() = default;

  explicit Grid(zisa::int_t _N_phys) {
    N_phys = _N_phys;
    N_phys_pad = 3 * N_phys / 2;
    N_fourier = N_phys / 2 + 1;
    N_fourier_pad = N_phys_pad / 2 + 1;
  }

  explicit Grid(zisa::int_t _N_phys, zisa::int_t _N_phys_pad) {
    N_phys = _N_phys;
    N_phys_pad = _N_phys_pad;
    N_fourier = N_phys / 2 + 1;
    N_fourier_pad = N_phys_pad / 2 + 1;
  }

  zisa::shape_t<dim_v + 1> shape_phys(zisa::int_t n_vars) const {
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    for (zisa::int_t i = 1; i < dim_v + 1; ++i) {
      shape[i] = N_phys;
    }
    return shape;
  }

#if AZEBAN_HAS_MPI
  zisa::shape_t<dim_v + 1> shape_phys(zisa::int_t n_vars,
                                      const Communicator *comm) const {
    const int rank = comm->rank();
    const int size = comm->size();
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    shape[1] = N_phys / size
               + (zisa::integer_cast<zisa::int_t>(rank) < N_phys % size);
    for (zisa::int_t i = 2; i < dim_v + 1; ++i) {
      shape[i] = N_phys;
    }
    return shape;
  }
#endif

  zisa::shape_t<dim_v + 1> shape_fourier(zisa::int_t n_vars) const {
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    for (zisa::int_t i = 1; i < dim_v; ++i) {
      shape[i] = N_phys;
    }
    shape[dim_v] = N_fourier;
    return shape;
  }

#if AZEBAN_HAS_MPI
  zisa::shape_t<dim_v + 1> shape_fourier(zisa::int_t n_vars,
                                         const Communicator *comm) const {
    const int rank = comm->rank();
    const int size = comm->size();
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    if (dim_v == 1) {
      shape[1] = N_fourier;
    } else if (dim_v == 2) {
      shape[1] = N_fourier / size
                 + (zisa::integer_cast<zisa::int_t>(rank) < N_fourier % size);
      shape[2] = N_phys;
    } else if (dim_v == 3) {
      shape[1] = N_fourier / size
                 + (zisa::integer_cast<zisa::int_t>(rank) < N_fourier % size);
      shape[2] = N_phys;
      shape[3] = N_phys;
    } else {
      LOG_ERR("Unsupported Dimension");
    }
    return shape;
  }
#endif

  zisa::shape_t<dim_v + 1> shape_phys_pad(zisa::int_t n_vars) const {
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    for (zisa::int_t i = 1; i < dim_v + 1; ++i) {
      shape[i] = N_phys_pad;
    }
    return shape;
  }

#if AZEBAN_HAS_MPI
  zisa::shape_t<dim_v + 1> shape_phys_pad(zisa::int_t n_vars,
                                          const Communicator *comm) const {
    const int rank = comm->rank();
    const int size = comm->size();
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    shape[1] = N_phys_pad / size
               + (zisa::integer_cast<zisa::int_t>(rank) < N_phys_pad % size);
    for (zisa::int_t i = 2; i < dim_v + 1; ++i) {
      shape[i] = N_phys_pad;
    }
    return shape;
  }
#endif

  zisa::shape_t<dim_v + 1> shape_fourier_pad(zisa::int_t n_vars) const {
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    for (zisa::int_t i = 1; i < dim_v; ++i) {
      shape[i] = N_phys_pad;
    }
    shape[dim_v] = N_fourier_pad;
    return shape;
  }

#if AZEBAN_HAS_MPI
  zisa::shape_t<dim_v + 1> shape_fourier_pad(zisa::int_t n_vars,
                                             const Communicator *comm) const {
    const int rank = comm->rank();
    const int size = comm->size();
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    if (dim_v == 1) {
      shape[1] = N_fourier_pad;
    } else if (dim_v == 2) {
      shape[1]
          = N_fourier_pad / size
            + (zisa::integer_cast<zisa::int_t>(rank) < N_fourier_pad % size);
      shape[2] = N_phys_pad;
    } else if (dim_v == 3) {
      shape[1]
          = N_fourier_pad / size
            + (zisa::integer_cast<zisa::int_t>(rank) < N_fourier_pad % size);
      shape[2] = N_phys_pad;
      shape[3] = N_phys_pad;
    } else {
      LOG_ERR("Unsupported Dimension");
    }
    return shape;
  }
#endif

  zisa::array<real_t, dim_v + 1>
  make_array_phys(zisa::int_t n_vars, zisa::device_type device) const {
    return zisa::array<real_t, dim_v + 1>(shape_phys(n_vars), device);
  }

#if AZEBAN_HAS_MPI
  zisa::array<real_t, dim_v + 1>
  make_array_phys(zisa::int_t n_vars,
                  zisa::device_type device,
                  const Communicator *comm) const {
    return zisa::array<real_t, dim_v + 1>(shape_phys(n_vars, comm), device);
  }
#endif

  zisa::array<complex_t, dim_v + 1>
  make_array_fourier(zisa::int_t n_vars, zisa::device_type device) const {
    return zisa::array<complex_t, dim_v + 1>(shape_fourier(n_vars), device);
  }

#if AZEBAN_HAS_MPI
  zisa::array<complex_t, dim_v + 1>
  make_array_fourier(zisa::int_t n_vars,
                     zisa::device_type device,
                     const Communicator *comm) const {
    return zisa::array<complex_t, dim_v + 1>(shape_fourier(n_vars, comm),
                                             device);
  }
#endif

  zisa::array<real_t, dim_v + 1>
  make_array_phys_pad(zisa::int_t n_vars, zisa::device_type device) const {
    return zisa::array<real_t, dim_v + 1>(shape_phys_pad(n_vars), device);
  }

#if AZEBAN_HAS_MPI
  zisa::array<real_t, dim_v + 1>
  make_array_phys_pad(zisa::int_t n_vars,
                      zisa::device_type device,
                      const Communicator *comm) const {
    return zisa::array<real_t, dim_v + 1>(shape_phys_pad(n_vars, comm), device);
  }
#endif

  zisa::array<complex_t, dim_v + 1>
  make_array_fourier_pad(zisa::int_t n_vars, zisa::device_type device) const {
    return zisa::array<complex_t, dim_v + 1>(shape_fourier_pad(n_vars), device);
  }

#if AZEBAN_HAS_MPI
  zisa::array<complex_t, dim_v + 1>
  make_array_fourier_pad(zisa::int_t n_vars,
                         zisa::device_type device,
                         const Communicator *comm) const {
    return zisa::array<complex_t, dim_v + 1>(shape_fourier_pad(n_vars, comm),
                                             device);
  }
#endif

#if AZEBAN_HAS_MPI
  zisa::int_t i_phys(zisa::int_t i, int rank, const Communicator *comm) const {
    const int size = comm->size();
    return i + rank * (N_phys / size)
           + zisa::min(zisa::integer_cast<zisa::int_t>(rank), N_phys % size);
  }

  zisa::int_t i_phys(zisa::int_t i, const Communicator *comm) const {
    const int rank = comm->rank();
    return i_phys(i, rank, comm);
  }

  zisa::int_t
  i_fourier(zisa::int_t i, int rank, const Communicator *comm) const {
    const int size = comm->size();
    if (dim_v == 1) {
      return i;
    } else if (dim_v == 2) {
      return i + rank * (N_fourier / size)
             + zisa::min(zisa::integer_cast<zisa::int_t>(rank),
                         N_fourier % size);
    } else if (dim_v == 3) {
      return i + rank * (N_fourier / size)
             + zisa::min(zisa::integer_cast<zisa::int_t>(rank),
                         N_fourier % size);
    } else {
      LOG_ERR("Unsupported Dimension");
    }
  }

  zisa::int_t i_fourier(zisa::int_t i, const Communicator *comm) const {
    const int rank = comm->rank();
    return i_fourier(i, rank, comm);
  }

  zisa::int_t j_phys(zisa::int_t j, int rank, const Communicator *comm) const {
    ZISA_UNUSED(comm);
    ZISA_UNUSED(rank);
    return j;
  }

  zisa::int_t j_phys(zisa::int_t j, const Communicator *comm) const {
    ZISA_UNUSED(comm);
    return j;
  }

  zisa::int_t
  j_fourier(zisa::int_t j, int rank, const Communicator *comm) const {
    ZISA_UNUSED(rank);
    ZISA_UNUSED(comm);
    return j;
  }

  zisa::int_t j_fourier(zisa::int_t j, const Communicator *comm) const {
    ZISA_UNUSED(comm);
    return j;
  }

  template <bool enable = Dim == 3,
            typename = typename std::enable_if<enable>::type>
  zisa::int_t k_phys(zisa::int_t k, int rank, const Communicator *comm) const {
    ZISA_UNUSED(rank);
    ZISA_UNUSED(comm);
    return k;
  }

  template <bool enable = Dim == 3,
            typename = typename std::enable_if<enable>::type>
  zisa::int_t k_phys(zisa::int_t k, const Communicator *comm) const {
    ZISA_UNUSED(comm);
    return k;
  }

  template <bool enable = Dim == 3,
            typename = typename std::enable_if<enable>::type>
  zisa::int_t
  k_fourier(zisa::int_t k, int rank, const Communicator *comm) const {
    ZISA_UNUSED(rank);
    ZISA_UNUSED(comm);
    return k;
  }

  template <bool enable = Dim == 3,
            typename = typename std::enable_if<enable>::type>
  zisa::int_t k_fourier(zisa::int_t k, const Communicator *comm) const {
    ZISA_UNUSED(comm);
    return k;
  }

  zisa::int_t
  i_phys_pad(zisa::int_t i, int rank, const Communicator *comm) const {
    const int size = comm->size();
    return i + rank * (N_phys_pad / size)
           + zisa::min(zisa::integer_cast<zisa::int_t>(rank),
                       N_phys_pad % size);
  }

  zisa::int_t i_phys_pad(zisa::int_t i, const Communicator *comm) const {
    const int rank = comm->rank();
    return i_phys_pad(i, rank, comm);
  }

  zisa::int_t
  i_fourier_pad(zisa::int_t i, int rank, const Communicator *comm) const {
    const int size = comm->size();
    if (dim_v == 1) {
      return i;
    } else if (dim_v == 2) {
      return i + rank * (N_fourier_pad / size)
             + zisa::min(zisa::integer_cast<zisa::int_t>(rank),
                         N_fourier_pad % size);
    } else if (dim_v == 3) {
      return i + rank * (N_fourier_pad / size)
             + zisa::min(zisa::integer_cast<zisa::int_t>(rank),
                         N_fourier_pad % size);
    } else {
      LOG_ERR("Unsupported Dimension");
    }
  }

  zisa::int_t i_fourier_pad(zisa::int_t i, const Communicator *comm) const {
    const int rank = comm->rank();
    return i_fourier_pad(i, rank, comm);
  }

  zisa::int_t
  j_phys_pad(zisa::int_t j, int rank, const Communicator *comm) const {
    ZISA_UNUSED(rank);
    ZISA_UNUSED(comm);
    return j;
  }

  zisa::int_t j_phys_pad(zisa::int_t j, const Communicator *comm) const {
    ZISA_UNUSED(comm);
    return j;
  }

  zisa::int_t
  j_fourier_pad(zisa::int_t j, int rank, const Communicator *comm) const {
    ZISA_UNUSED(rank);
    ZISA_UNUSED(comm);
    return j;
  }

  zisa::int_t j_fourier_pad(zisa::int_t j, const Communicator *comm) const {
    ZISA_UNUSED(comm);
    return j;
  }

  template <bool enable = Dim == 3,
            typename = typename std::enable_if<enable>::type>
  zisa::int_t
  k_phys_pad(zisa::int_t k, int rank, const Communicator *comm) const {
    ZISA_UNUSED(rank);
    ZISA_UNUSED(comm);
    return k;
  }

  template <bool enable = Dim == 3,
            typename = typename std::enable_if<enable>::type>
  zisa::int_t k_phys_pad(zisa::int_t k, const Communicator *comm) const {
    ZISA_UNUSED(comm);
    return k;
  }

  template <bool enable = Dim == 3,
            typename = typename std::enable_if<enable>::type>
  zisa::int_t
  k_fourier_pad(zisa::int_t k, int rank, const Communicator *comm) const {
    ZISA_UNUSED(rank);
    ZISA_UNUSED(comm);
    return k;
  }

  template <bool enable = Dim == 3,
            typename = typename std::enable_if<enable>::type>
  zisa::int_t k_fourier_pad(zisa::int_t k, const Communicator *comm) const {
    ZISA_UNUSED(comm);
    return k;
  }
#endif
};

}

#endif
