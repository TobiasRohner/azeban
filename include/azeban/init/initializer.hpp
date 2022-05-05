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
#ifndef AZEBAN_INIT_INITIALIZER_HPP_
#define AZEBAN_INIT_INITIALIZER_HPP_

#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <azeban/operations/fft_factory.hpp>
#include <azeban/operations/leray.hpp>
#include <azeban/operations/scale.hpp>
#include <fmt/core.h>
#include <vector>
#include <zisa/memory/array_view.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/communicator.hpp>
#include <azeban/operations/fft_mpi_factory.hpp>
#endif

namespace azeban {

template <int Dim>
class Initializer {
public:
  Initializer() = default;
  Initializer(const Initializer &) = default;
  Initializer(Initializer &&) = default;

  virtual ~Initializer() = default;

  Initializer &operator=(const Initializer &) = default;
  Initializer &operator=(Initializer &&) = default;

  virtual void initialize(const zisa::array_view<real_t, Dim + 1> &u) {
    if constexpr (Dim > 1) {
      zisa::shape_t<Dim + 1> u_hat_shape = u.shape();
      u_hat_shape[Dim] = u.shape(Dim) / 2 + 1;
      auto u_hat
          = zisa::array<complex_t, Dim + 1>(u_hat_shape, u.memory_location());

      auto fft = make_fft<Dim>(u_hat, u);

      do_initialize(u);
      fft->forward();
      leray(u_hat);
      scale(complex_t(u.shape(0)) / zisa::product(u.shape()),
            zisa::array_view<complex_t, Dim + 1>(u_hat));
      fft->backward();
    } else {
      do_initialize(u);
    }
  }

  virtual void initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat) {
    if constexpr (Dim > 1) {
      do_initialize(u_hat);
      leray(u_hat);
    } else {
      do_initialize(u_hat);
    }
  }

#if AZEBAN_HAS_MPI
  virtual void initialize(const zisa::array_view<real_t, Dim + 1> &u,
                          const Grid<Dim> &grid,
                          const Communicator *comm,
                          void *work_area = nullptr) {
    if constexpr (Dim > 1) {
      auto init = [&](const zisa::array_view<real_t, Dim + 1> &u_device) {
        auto u_hat = grid.make_array_fourier(
            u.shape(0), zisa::device_type::cuda, comm);
        auto fft
            = make_fft_mpi<Dim>(u_hat, u_device, comm, FFT_BACKWARD, work_area);
        do_initialize(u_hat, grid, comm, work_area);
        leray(u_hat, grid, comm);
        scale(complex_t(1) / zisa::pow<Dim>(grid.N_phys), u_hat.view());
        fft->backward();
      };
      if (u.memory_location() == zisa::device_type::cpu) {
        auto u_device
            = zisa::array<real_t, Dim + 1>(u.shape(), zisa::device_type::cuda);
        init(u_device);
        zisa::copy(u, u_device);
      } else if (u.memory_location() == zisa::device_type::cuda) {
        init(u);
      } else {
        LOG_ERR("Unknown memory_location");
      }
    } else {
      ZISA_UNUSED(comm);
      ZISA_UNUSED(work_area);
      LOG_ERR("Unsupported dimension");
    }
  }

  virtual void initialize(const zisa::array_view<complex_t, Dim + 1> &u,
                          const Grid<Dim> &grid,
                          const Communicator *comm,
                          void *work_area = nullptr) {
    if constexpr (Dim > 1) {
      do_initialize(u, grid, comm, work_area);
      leray(u, grid, comm);
    } else {
      ZISA_UNUSED(comm);
      ZISA_UNUSED(work_area);
      LOG_ERR("Unsupported dimension");
    }
  }
#endif

protected:
  virtual void do_initialize(const zisa::array_view<real_t, Dim + 1> &u) = 0;
  virtual void do_initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat)
      = 0;

#if AZEBAN_HAS_MPI
  virtual void do_initialize(const zisa::array_view<real_t, Dim + 1> &u,
                             const Grid<Dim> &grid,
                             const Communicator *comm,
                             void *) {
    if constexpr (Dim > 1) {
      auto init = [&](const zisa::array_view<real_t, Dim + 1> &u_host) {
        const int rank = comm->rank();
        const int size = comm->size();
        zisa::array<real_t, Dim + 1> u_init;
        if (rank == 0) {
          u_init
              = grid.make_array_phys(u_host.shape(0), zisa::device_type::cpu);
          do_initialize(u_init);
        }
        std::vector<int> cnts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
          cnts[r]
              = zisa::pow<Dim - 1>(grid.N_phys)
                * (grid.N_phys / size
                   + (zisa::integer_cast<zisa::int_t>(r) < grid.N_phys % size));
        }
        displs[0] = 0;
        for (int r = 1; r < size; ++r) {
          displs[r] = displs[r - 1] + cnts[r - 1];
        }
        std::vector<MPI_Request> reqs(u_host.shape(0));
        const zisa::int_t n_elems_per_component_glob
            = zisa::product(grid.shape_phys(1));
        const zisa::int_t n_elems_per_component_loc
            = zisa::product(grid.shape_phys(1, comm));
        for (zisa::int_t i = 0; i < u_host.shape(0); ++i) {
          MPI_Iscatterv(u_init.raw() + i * n_elems_per_component_glob,
                        cnts.data(),
                        displs.data(),
                        mpi_type<real_t>(),
                        u_host.raw() + i * n_elems_per_component_loc,
                        cnts[rank],
                        mpi_type<real_t>(),
                        0,
                        comm->get_mpi_comm(),
                        &reqs[i]);
        }
        MPI_Waitall(u_host.shape(0), reqs.data(), MPI_STATUSES_IGNORE);
      };
      if (u.memory_location() == zisa::device_type::cpu) {
        init(u);
      }
#if ZISA_HAS_CUDA
      else if (u.memory_location() == zisa::device_type::cuda) {
        auto u_host
            = zisa::array<real_t, Dim + 1>(u.shape(), zisa::device_type::cpu);
        init(u_host);
        zisa::copy(u, u_host);
      }
#endif
      else {
        LOG_ERR("Unknown memory location");
      }
    } else {
      ZISA_UNUSED(comm);
      LOG_ERR("Unsupported Dimension");
    }
  }

  virtual void do_initialize(const zisa::array_view<complex_t, Dim + 1> &u,
                             const Grid<Dim> &grid,
                             const Communicator *comm,
                             void *work_area = nullptr) {
    if constexpr (Dim > 1) {
      auto u_device
          = grid.make_array_phys(u.shape(0), zisa::device_type::cuda, comm);
      auto u_hat_device
          = grid.make_array_fourier(u.shape(0), zisa::device_type::cuda, comm);
      auto fft = make_fft_mpi<Dim>(
          u_hat_device, u_device, comm, FFT_FORWARD, work_area);
      do_initialize(u_device, grid, comm, work_area);
      fft->forward();
      zisa::copy(u, u_hat_device);
    } else {
      ZISA_UNUSED(comm);
      ZISA_UNUSED(work_area);
      LOG_ERR("Unsupported Dimension");
    }
  }
#endif
};

}

#endif
