#ifndef FORWARD_EULER_H_
#define FORWARD_EULER_H_

#include "time_integrator.hpp"
#include <azeban/operations/axpy.hpp>
#include <azeban/profiler.hpp>

namespace azeban {

template <int Dim>
class ForwardEuler final : public TimeIntegrator<Dim> {
  using super = TimeIntegrator<Dim>;

public:
  static constexpr int dim_v = Dim;

  ForwardEuler() = delete;
  ForwardEuler(zisa::device_type device,
               const zisa::shape_t<dim_v + 1> &shape,
               const std::shared_ptr<Equation<dim_v>> &equation)
      : super(device, equation), dudt_(shape, device) {}
  ForwardEuler(const ForwardEuler &) = delete;
  ForwardEuler(ForwardEuler &&) = default;

  virtual ~ForwardEuler() override = default;

  ForwardEuler &operator=(const ForwardEuler &) = delete;
  ForwardEuler &operator=(ForwardEuler &&) = default;

  virtual void
  integrate(real_t dt,
            const zisa::array_view<complex_t, dim_v + 1> &u) override {
    AZEBAN_PROFILE_START("forward_euler::integrate");
    zisa::internal::copy(dudt_.raw(),
                         dudt_.device(),
                         u.raw(),
                         u.memory_location(),
                         zisa::product(dudt_.shape()));
    equation_->dudt(dudt_);
    axpy(complex_t(dt), zisa::array_const_view<complex_t, dim_v + 1>(dudt_), u);
    AZEBAN_PROFILE_STOP("forward_euler::integrate");
  }

  using super::equation;
  using super::memory_location;

protected:
  using super::device_;
  using super::equation_;

private:
  zisa::array<complex_t, dim_v + 1> dudt_;
};

}

#endif
