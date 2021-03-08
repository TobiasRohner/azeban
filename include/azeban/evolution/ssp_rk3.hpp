#ifndef SSP_RK3_H_
#define SSP_RK3_H_

#include "time_integrator.hpp"
#include <azeban/operations/axpby.hpp>
#include <azeban/operations/axpy.hpp>

namespace azeban {

template <typename Scalar, int Dim>
class SSP_RK3 : public TimeIntegrator<Scalar, Dim> {
  using super = TimeIntegrator<Scalar, Dim>;

public:
  using scalar_t = Scalar;
  static constexpr int dim_v = Dim;

  SSP_RK3() = delete;
  SSP_RK3(zisa::device_type device,
          const zisa::shape_t<dim_v + 1> &shape,
          const std::shared_ptr<Equation<scalar_t, dim_v>> &equation)
      : super(device, equation), u1_(shape, device), u2_(shape, device) {}
  SSP_RK3(const SSP_RK3 &) = delete;
  SSP_RK3(SSP_RK3 &&) = default;

  virtual ~SSP_RK3() override = default;

  SSP_RK3 &operator=(const SSP_RK3 &) = delete;
  SSP_RK3 &operator=(SSP_RK3 &&) = default;

  virtual void
  integrate(real_t dt,
            const zisa::array_view<scalar_t, dim_v + 1> &u) override {
    zisa::copy(u1_, u);
    equation_->dudt(u1_);
    axpby<complex_t, dim_v + 1>(1, u, dt, u1_);
    zisa::copy(u2_, u1_);
    equation_->dudt(u2_);
    axpby<complex_t, dim_v + 1>(1, u1_, dt, u2_);
    axpby<complex_t, dim_v + 1>(3. / 4, u, 1. / 4, u2_);
    zisa::copy(u1_, u2_);
    equation_->dudt(u1_);
    axpby<complex_t, dim_v + 1>(1, u2_, dt, u1_);
    axpby<complex_t, dim_v + 1>(2. / 3, u1_, 1. / 3, u);
  }

  using super::equation;
  using super::memory_location;

protected:
  using super::device_;
  using super::equation_;

private:
  zisa::array<scalar_t, dim_v + 1> u1_;
  zisa::array<scalar_t, dim_v + 1> u2_;
};

}

#endif
