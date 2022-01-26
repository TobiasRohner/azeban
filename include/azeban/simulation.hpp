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
#include <fmt/core.h>
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif

namespace azeban {

template <int Dim>
class Simulation {
public:
  static constexpr int dim_v = Dim;

  Simulation() = delete;
  Simulation(const zisa::shape_t<dim_v + 1> &shape,
             const CFL<Dim> &cfl,
             const std::shared_ptr<TimeIntegrator<dim_v>> &timestepper,
             zisa::device_type device = zisa::device_type::cpu)
      : u_(shape, device),
        u_view_(u_.shape(), u_.raw(), u_.device()),
        cfl_(cfl),
        timestepper_(timestepper),
        time_(0),
        memory_location_(device) {
    zisa::shape_t<dim_v + 1> u_view_shape = u_.shape();
    u_view_shape[0] = dim_v;
    u_view_ = zisa::array_view<complex_t, dim_v + 1>(
        u_view_shape, u_.raw(), u_.device());
  }
  Simulation(const zisa::array_const_view<complex_t, dim_v + 1> &u,
             const CFL<Dim> cfl,
             const std::shared_ptr<TimeIntegrator<dim_v>> &timestepper)
      : u_(u.shape(), u.memory_location()),
        u_view_(u_.shape(), u_.raw(), u_.device()),
        cfl_(cfl),
        timestepper_(timestepper),
        time_(0),
        memory_location_(u.memory_location()) {
    zisa::copy(u_, u);
    zisa::shape_t<dim_v + 1> u_view_shape = u_.shape();
    u_view_shape[0] = dim_v;
    u_view_ = zisa::array_view<complex_t, dim_v + 1>(
        u_view_shape, u_.raw(), u_.device());
  }
  Simulation(const Simulation &) = delete;
  Simulation(Simulation &&) = default;

  Simulation &operator=(const Simulation &) = delete;
  Simulation &operator=(Simulation &&) = default;

  void simulate_until(real_t t) {
    real_t dt = cfl_.dt(u_view_);
    while (time_ < t - dt) {
      if (dt <= 1e-10) {
        fmt::print(stderr, "Warning: Timestep is tiny. dt = {}\n", dt);
      }
      timestepper_->integrate(dt, u_);
      time_ += dt;
      dt = cfl_.dt(u_view_);
    }
    timestepper_->integrate(t - time_, u_);
    time_ = t;
  }

  void simulate_for(real_t t) { simulate_until(time_ + t); }

  real_t step() {
    const real_t dt = cfl_.dt(u_view_);
    timestepper_->integrate(dt, u_);
    time_ += dt;
    return dt;
  }

#if AZEBAN_HAS_MPI
  void simulate_until(real_t t, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    real_t dt = cfl_.dt(u_view_, comm);
    while (time_ < t - dt) {
      if (rank == 0 && dt <= 1e-10) {
        fmt::print(stderr, "Warning: Timestep is tiny. dt = {}\n", dt);
      }
      timestepper_->integrate(dt, u_);
      time_ += dt;
      dt = cfl_.dt(u_view_, comm);
    }
    timestepper_->integrate(t - time_, u_);
    time_ = t;
  }

  void simulate_for(real_t t, MPI_Comm comm) {
    simulate_until(time_ + t, comm);
  }

  real_t step(MPI_Comm comm) {
    const real_t dt = cfl_.dt(u_view_, comm);
    timestepper_->integrate(dt, u_);
    time_ += dt;
    return dt;
  }
#endif

  void reset() { time_ = 0; }

  real_t time() const { return time_; }
  zisa::array_view<complex_t, dim_v + 1> u() { return u_; }
  zisa::array_const_view<complex_t, dim_v + 1> u() const { return u_; }
  const Grid<dim_v> &grid() const { return cfl_.grid(); }
  zisa::int_t n_vars() const { return u_.shape(0); }
  zisa::device_type memory_location() const { return memory_location_; }
  std::shared_ptr<const Equation<dim_v>> equation() const {
    return timestepper_->equation();
  }

private:
  zisa::array<complex_t, dim_v + 1> u_;
  zisa::array_view<complex_t, dim_v + 1> u_view_;
  CFL<Dim> cfl_;
  std::shared_ptr<TimeIntegrator<dim_v>> timestepper_;
  real_t time_;
  zisa::device_type memory_location_;
};

}

#endif
