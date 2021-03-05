#ifndef SSP_RK2_H_
#define SSP_RK2_H_

#include "time_integrator.hpp"
#include <azeban/operations/axpy.hpp>

namespace azeban {

template <typename Scalar, int Dim>
class SSP_RK2 final : public TimeIntegrator<Scalar, Dim> {
  using super = TimeIntegrator<Scalar, Dim>;

public:
  using scalar_t = Scalar;
  static constexpr int dim_v = Dim;

  SSP_RK2() = delete;
  SSP_RK2(zisa::device_type device,
          const zisa::shape_t<dim_v + 1> &shape,
          const std::shared_ptr<Equation<scalar_t, dim_v>> &equation)
      : super(device, equation), u_star_(shape, device), dudt_(shape, device) {}
  SSP_RK2(const SSP_RK2 &) = delete;
  SSP_RK2(SSP_RK2 &&) = default;

  virtual ~SSP_RK2() override = default;

  SSP_RK2 &operator=(const SSP_RK2 &) = delete;
  SSP_RK2 &operator=(SSP_RK2 &&) = default;

  virtual void
  integrate(real_t dt,
            const zisa::array_view<scalar_t, dim_v + 1> &u) override {
    zisa::copy(dudt_, u);
    zisa::copy(u_star_, u);
    equation_->dudt(dudt_);
    axpy(scalar_t(0.5 * dt),
         zisa::array_const_view<scalar_t, dim_v + 1>(dudt_),
         u);
    axpy(scalar_t(dt),
         zisa::array_const_view<scalar_t, dim_v + 1>(dudt_),
         zisa::array_view<scalar_t, dim_v + 1>(u_star_));
    equation_->dudt(u_star_);
    axpy(scalar_t(0.5 * dt),
         zisa::array_const_view<scalar_t, dim_v + 1>(u_star_),
         u);
  }

protected:
  using super::equation;
  using super::memory_location;

private:
  using super::device_;
  using super::equation_;
  zisa::array<scalar_t, dim_v + 1> u_star_;
  zisa::array<scalar_t, dim_v + 1> dudt_;
};

}

#endif
