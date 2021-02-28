#ifndef SIMULATION_H_
#define SIMULATION_H_

#include <azeban/evolution/cfl.hpp>
#include <azeban/equations/equation.hpp>
#include <azeban/evolution/time_integrator.hpp>



namespace azeban {


template<typename Scalar, int Dim>
class Simulation {
public:
  using scalar_t = Scalar;
  static constexpr int dim_v = Dim;

  Simulation() = delete;
  Simulation(const zisa::shape_t<dim_v> &shape,
	     const CFL &cfl,
	     const std::shared_ptr<TimeIntegrator<scalar_t, dim_v>> &timestepper,
	     zisa::device_type device = zisa::device_type::cpu)
      : u_(shape, device),
	cfl_(cfl),
	timestepper_(timestepper),
	time_(0) { }
  Simulation(const zisa::array_const_view<scalar_t, dim_v> &u,
	     const CFL cfl,
	     const std::shared_ptr<TimeIntegrator<scalar_t, dim_v>> &timestepper)
      : u_(u.shape(), u.memory_location()),
	cfl_(cfl),
	timestepper_(timestepper),
	time_(0) {
    // Ugly, but normal copy doesn't work for some reason
    zisa::internal::copy(u_.raw(), u_.device(),
			 u.raw(), u.memory_location(),
			 zisa::product(u_.shape()));
  }
  Simulation(const Simulation&) = delete;
  Simulation(Simulation&&) = default;

  Simulation& operator=(const Simulation&) = delete;
  Simulation& operator=(Simulation&&) = default;

  void simulate_until(real_t t) {
    real_t dt = cfl_.dt(zisa::array_const_view<scalar_t, Dim>(u_));
    while (time_ < t-dt) {
      timestepper_->integrate(dt, u_);
      time_ += dt;
      dt = cfl_.dt(zisa::array_const_view<scalar_t, Dim>(u_));
    }
    timestepper_->integrate(t-time_, u_);
    time_ = t;
  }

  void simulate_for(real_t t) {
    simulate_until(time_ + t);
  }

  real_t time() const { return time_; }
  zisa::array_view<scalar_t, dim_v> u() { return u_; }
  zisa::array_const_view<scalar_t, dim_v> u() const { return u_; }

private:
  zisa::array<scalar_t, dim_v> u_;
  CFL cfl_;
  std::shared_ptr<TimeIntegrator<scalar_t, dim_v>> timestepper_;
  real_t time_;
};


}



#endif
