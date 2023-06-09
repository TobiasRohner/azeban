#include <azeban/simulation.hpp>
#include <fmt/core.h>

namespace azeban {

template <int Dim>
Simulation<Dim>::Simulation(
    const Grid<Dim> &grid,
    real_t C,
    const std::shared_ptr<TimeIntegrator<dim_v>> &timestepper,
    zisa::device_type device)
    : u_(grid.shape_fourier(timestepper->equation()->n_vars()), device),
      u_view_(u_.shape(), u_.raw(), u_.device()),
      C_(C),
      grid_(grid),
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
    const Grid<Dim> &grid,
    real_t C,
    const std::shared_ptr<TimeIntegrator<dim_v>> &timestepper)
    : u_(u.shape(), u.memory_location()),
      u_view_(u_.shape(), u_.raw(), u_.device()),
      C_(C),
      grid_(grid),
      timestepper_(timestepper),
      time_(0),
      memory_location_(u.memory_location()) {
  zisa::copy(u_, u);
  zisa::shape_t<dim_v + 1> u_view_shape = u_.shape();
  u_view_shape[0] = dim_v;
  u_view_ = zisa::array_view<complex_t, dim_v + 1>(
      u_view_shape, u_.raw(), u_.device());
}

#if AZEBAN_HAS_MPI
template <int Dim>
Simulation<Dim>::Simulation(
    const Grid<Dim> &grid,
    real_t C,
    const std::shared_ptr<TimeIntegrator<dim_v>> &timestepper,
    zisa::device_type device,
    const Communicator *comm)
    : u_(grid.shape_fourier(timestepper->equation()->n_vars(), comm), device),
      u_view_(u_.shape(), u_.raw(), u_.device()),
      C_(C),
      grid_(grid),
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
    const Grid<Dim> &grid,
    real_t C,
    const std::shared_ptr<TimeIntegrator<dim_v>> &timestepper,
    const Communicator *comm)
    : u_(u.shape(), u.memory_location()),
      u_view_(u_.shape(), u_.raw(), u_.device()),
      C_(C),
      grid_(grid),
      timestepper_(timestepper),
      time_(0),
      memory_location_(u.memory_location()) {
  ZISA_UNUSED(comm);
  zisa::copy(u_, u);
  zisa::shape_t<dim_v + 1> u_view_shape = u_.shape();
  u_view_shape[0] = dim_v;
  u_view_ = zisa::array_view<complex_t, dim_v + 1>(
      u_view_shape, u_.raw(), u_.device());
}
#endif

template <int Dim>
void Simulation<Dim>::simulate_until(real_t t) {
  while (time_ < t) {
    const real_t eps = equation()->visc();
    const real_t max_dt = zisa::min(
        t - time_,
        real_t(C_ * 2. / (eps * zisa::pow<2>(zisa::pi * grid_.N_phys))));
    const real_t dt = timestepper_->integrate(t, max_dt, C_, u_);
    if (dt <= 1e-10) {
      fmt::print(stderr, "Warning: Timestep is tiny. dt = {}\n", dt);
    }
    time_ += dt;
  }
  time_ = t;
}

template <int Dim>
void Simulation<Dim>::simulate_for(real_t t) {
  simulate_until(time_ + t);
}

#if AZEBAN_HAS_MPI
template <int Dim>
void Simulation<Dim>::simulate_until(real_t t, const Communicator *comm) {
  const int rank = comm->rank();
  while (time_ < t) {
    const real_t eps = equation()->visc();
    const real_t max_dt = zisa::min(
        t - time_,
        real_t(C_ * 2. / (eps * zisa::pow<2>(zisa::pi * grid_.N_phys))));
    const real_t dt = timestepper_->integrate(t, max_dt, C_, u_);
    if (rank == 0 && dt <= 1e-10) {
      fmt::print(stderr, "Warning: Timestep is tiny. dt = {}\n", dt);
    }
    time_ += dt;
  }
}

template <int Dim>
void Simulation<Dim>::simulate_for(real_t t, const Communicator *comm) {
  simulate_until(time_ + t, comm);
}
#endif

template class Simulation<1>;
template class Simulation<2>;
template class Simulation<3>;

}
