#ifndef SIMULATION_H_
#define SIMULATION_H_

#include <azeban/equations/equation.hpp>
#include <azeban/evolution/cfl.hpp>
#include <azeban/evolution/time_integrator.hpp>
#include <fmt/core.h>

namespace azeban {

template <typename Scalar, int Dim>
class Simulation {
public:
  using scalar_t = Scalar;
  static constexpr int dim_v = Dim;

  Simulation() = delete;
  Simulation(
      const zisa::shape_t<dim_v + 1> &shape,
      const CFL<Dim> &cfl,
      const std::shared_ptr<TimeIntegrator<scalar_t, dim_v>> &timestepper,
      zisa::device_type device = zisa::device_type::cpu)
      : u_(shape, device),
        cfl_(cfl),
        timestepper_(timestepper),
        time_(0),
        memory_location_(device) {}
  Simulation(
      const zisa::array_const_view<scalar_t, dim_v + 1> &u,
      const CFL<Dim> cfl,
      const std::shared_ptr<TimeIntegrator<scalar_t, dim_v>> &timestepper)
      : u_(u.shape(), u.memory_location()),
        cfl_(cfl),
        timestepper_(timestepper),
        time_(0),
        memory_location_(u.memory_location()) {
    zisa::copy(u_, u);
  }
  Simulation(const Simulation &) = delete;
  Simulation(Simulation &&) = default;

  Simulation &operator=(const Simulation &) = delete;
  Simulation &operator=(Simulation &&) = default;

  void simulate_until(real_t t) {
    real_t dt = cfl_.dt(u_);
    while (time_ < t - dt) {
      timestepper_->integrate(dt, u_);
      time_ += dt;
      dt = cfl_.dt(u_);
    }
    timestepper_->integrate(t - time_, u_);
    time_ = t;
  }

  void simulate_for(real_t t) { simulate_until(time_ + t); }

  real_t step() {
    const real_t dt = cfl_.dt(u_);
    timestepper_->integrate(dt, u_);
    time_ += dt;
    return dt;
  }

  real_t time() const { return time_; }
  zisa::array_view<scalar_t, dim_v + 1> u() { return u_; }
  zisa::array_const_view<scalar_t, dim_v + 1> u() const { return u_; }
  const Grid<dim_v> &grid() const { return cfl_.grid(); }
  zisa::int_t n_vars() const { return u_.shape(0); }
  zisa::device_type memory_location() const { return memory_location_; }

private:
  zisa::array<scalar_t, dim_v + 1> u_;
  CFL<Dim> cfl_;
  std::shared_ptr<TimeIntegrator<scalar_t, dim_v>> timestepper_;
  real_t time_;
  zisa::device_type memory_location_;
};

}

#endif
