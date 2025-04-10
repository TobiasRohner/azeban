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
#ifndef SIMULATION_H_
#define SIMULATION_H_

#include <azeban/equations/equation.hpp>
#include <azeban/evolution/cfl.hpp>
#include <azeban/evolution/time_integrator.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/communicator.hpp>
#endif

namespace azeban {

template <int Dim>
class Simulation {
public:
  static constexpr int dim_v = Dim;

  Simulation() = delete;
  Simulation(const Grid<Dim> &grid,
             double C,
             const std::shared_ptr<TimeIntegrator<dim_v>> &timestepper,
             zisa::device_type device = zisa::device_type::cpu);
  Simulation(const zisa::array_const_view<complex_t, dim_v + 1> &u,
             const Grid<Dim> &grid_,
             double C,
             const std::shared_ptr<TimeIntegrator<dim_v>> &timestepper);
#if AZEBAN_HAS_MPI
  Simulation(const Grid<Dim> &grid,
             double C,
             const std::shared_ptr<TimeIntegrator<dim_v>> &timestepper,
             zisa::device_type device,
             const Communicator *comm);
  Simulation(const zisa::array_const_view<complex_t, dim_v + 1> &u,
             const Grid<Dim> &grid_,
             double C,
             const std::shared_ptr<TimeIntegrator<dim_v>> &timestepper,
             const Communicator *comm);
#endif
  Simulation(const Simulation &) = delete;
  Simulation(Simulation &&) = default;

  Simulation &operator=(const Simulation &) = delete;
  Simulation &operator=(Simulation &&) = default;

  void simulate_until(double t);
  void simulate_for(double t);
#if AZEBAN_HAS_MPI
  void simulate_until(double t, const Communicator *comm);
  void simulate_for(double t, const Communicator *comm);
#endif

  void reset() { time_ = 0; }
  void set_time(double t) { time_ = t; }

  double time() const { return time_; }
  zisa::array_view<complex_t, dim_v + 1> u() { return u_; }
  zisa::array_const_view<complex_t, dim_v + 1> u() const { return u_; }
  const Grid<dim_v> &grid() const { return grid_; }
  zisa::int_t n_vars() const { return u_.shape(0); }
  zisa::device_type memory_location() const { return memory_location_; }
  std::shared_ptr<Equation<dim_v>> equation() {
    return timestepper_->equation();
  }
  std::shared_ptr<const Equation<dim_v>> equation() const {
    return timestepper_->equation();
  }

private:
  zisa::array<complex_t, dim_v + 1> u_;
  zisa::array_view<complex_t, dim_v + 1> u_view_;
  double C_;
  Grid<Dim> grid_;
  std::shared_ptr<TimeIntegrator<dim_v>> timestepper_;
  double time_;
  zisa::device_type memory_location_;
};

}

#endif
