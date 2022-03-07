#include <azeban/simulation.hpp>
#include <fmt/core.h>

namespace azeban {

template <int Dim>
Simulation<Dim>::Simulation(
    const zisa::shape_t<dim_v + 1> &shape,
    const CFL<Dim> &cfl,
    const std::shared_ptr<TimeIntegrator<dim_v>> &timestepper,
    zisa::device_type device)
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

template <int Dim>
Simulation<Dim>::Simulation(
    const zisa::array_const_view<complex_t, dim_v + 1> &u,
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

template <int Dim>
void Simulation<Dim>::simulate_until(real_t t) {
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

template <int Dim>
void Simulation<Dim>::simulate_for(real_t t) {
  simulate_until(time_ + t);
}

template <int Dim>
real_t Simulation<Dim>::step() {
  const real_t dt = cfl_.dt(u_view_);
  timestepper_->integrate(dt, u_);
  time_ += dt;
  return dt;
}

#if AZEBAN_HAS_MPI
template <int Dim>
void Simulation<Dim>::simulate_until(real_t t, const Communicator *comm) {
  const int rank = comm->rank();
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

template <int Dim>
void Simulation<Dim>::simulate_for(real_t t, const Communicator *comm) {
  simulate_until(time_ + t, comm);
}

template <int Dim>
real_t Simulation<Dim>::step(const Communicator *comm) {
  const real_t dt = cfl_.dt(u_view_, comm);
  timestepper_->integrate(dt, u_);
  time_ += dt;
  return dt;
}
#endif

template class Simulation<1>;
template class Simulation<2>;
template class Simulation<3>;

}
