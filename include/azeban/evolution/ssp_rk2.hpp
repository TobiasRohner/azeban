#ifndef SSP_RK2_H_
#define SSP_RK2_H_

#include "time_integrator.hpp"
#include <azeban/operations/axpy.hpp>
#include <azeban/profiler.hpp>

namespace azeban {

template <int Dim>
class SSP_RK2 final : public TimeIntegrator<Dim> {
  using super = TimeIntegrator<Dim>;

public:
  static constexpr int dim_v = Dim;

  SSP_RK2() = delete;
  SSP_RK2(zisa::device_type device,
          const zisa::shape_t<dim_v + 1> &shape,
          const std::shared_ptr<Equation<dim_v>> &equation)
      : super(device, equation), u_star_(shape, device), dudt_(shape, device) {}
  SSP_RK2(const SSP_RK2 &) = delete;
  SSP_RK2(SSP_RK2 &&) = default;

  virtual ~SSP_RK2() override = default;

  SSP_RK2 &operator=(const SSP_RK2 &) = delete;
  SSP_RK2 &operator=(SSP_RK2 &&) = default;

  virtual void
  integrate(real_t dt,
            const zisa::array_view<complex_t, dim_v + 1> &u) override {
    AZEBAN_PROFILE_START("SSP_RK2::integrate");
    zisa::copy(dudt_, u);
    zisa::copy(u_star_, u);
    equation_->dudt(dudt_);
    axpy(complex_t(0.5 * dt),
         zisa::array_const_view<complex_t, dim_v + 1>(dudt_),
         u);
    axpy(complex_t(dt),
         zisa::array_const_view<complex_t, dim_v + 1>(dudt_),
         zisa::array_view<complex_t, dim_v + 1>(u_star_));
    equation_->dudt(u_star_);
    axpy(complex_t(0.5 * dt),
         zisa::array_const_view<complex_t, dim_v + 1>(u_star_),
         u);
    AZEBAN_PROFILE_STOP("SSP_RK2::integrate");
  }

  using super::equation;
  using super::memory_location;

protected:
  using super::device_;
  using super::equation_;

private:
  zisa::array<complex_t, dim_v + 1> u_star_;
  zisa::array<complex_t, dim_v + 1> dudt_;
};

}

#endif
