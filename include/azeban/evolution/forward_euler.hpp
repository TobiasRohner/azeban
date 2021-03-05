#ifndef FORWARD_EULER_H_
#define FORWARD_EULER_H_

#include "time_integrator.hpp"
#include <azeban/operations/axpy.hpp>

namespace azeban {

template <typename Scalar, int Dim>
class ForwardEuler final : public TimeIntegrator<Scalar, Dim> {
  using super = TimeIntegrator<Scalar, Dim>;

public:
  using scalar_t = Scalar;
  static constexpr int dim_v = Dim;

  ForwardEuler() = delete;
  ForwardEuler(zisa::device_type device,
               const zisa::shape_t<dim_v> &shape,
               const std::shared_ptr<Equation<scalar_t, dim_v>> &equation)
      : super(device, equation), dudt_(shape, device) {}
  ForwardEuler(const ForwardEuler &) = delete;
  ForwardEuler(ForwardEuler &&) = default;

  virtual ~ForwardEuler() override = default;

  ForwardEuler &operator=(const ForwardEuler &) = delete;
  ForwardEuler &operator=(ForwardEuler &&) = default;

  virtual void integrate(real_t dt,
                         const zisa::array_view<scalar_t, dim_v+1> &u) override {
    zisa::internal::copy(dudt_.raw(),
                         dudt_.device(),
                         u.raw(),
                         u.memory_location(),
                         zisa::product(dudt_.shape()));
    equation_->dudt(dudt_);
    axpy(scalar_t(dt), zisa::array_const_view<scalar_t, dim_v+1>(dudt_), u);
  }

  using super::equation;
  using super::memory_location;

protected:
  using super::device_;
  using super::equation_;

private:
  zisa::array<scalar_t, dim_v+1> dudt_;
};

}

#endif
