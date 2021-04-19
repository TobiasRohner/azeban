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
    real_t dt = cfl_.dt(u_view_, comm);
    while (time_ < t - dt) {
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

  real_t time() const { return time_; }
  zisa::array_view<complex_t, dim_v + 1> u() { return u_; }
  zisa::array_const_view<complex_t, dim_v + 1> u() const { return u_; }
  const Grid<dim_v> &grid() const { return cfl_.grid(); }
  zisa::int_t n_vars() const { return u_.shape(0); }
  zisa::device_type memory_location() const { return memory_location_; }

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
