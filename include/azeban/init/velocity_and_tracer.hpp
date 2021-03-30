#ifndef VELOCITY_AND_TRACER_H_
#define VELOCITY_AND_TRACER_H_

#include "initializer.hpp"

namespace azeban {

template <int Dim>
class VelocityAndTracer final : public Initializer<Dim> {
  using super = Initializer<Dim>;
  static constexpr int dim_v = Dim;

public:
  VelocityAndTracer(const std::shared_ptr<Initializer<Dim>> &init_u,
                    const std::shared_ptr<Initializer<Dim>> &init_rho)
      : init_u_(init_u), init_rho_(init_rho) {}
  VelocityAndTracer(const VelocityAndTracer &) = default;
  VelocityAndTracer(VelocityAndTracer &&) = default;

  virtual ~VelocityAndTracer() override = default;

  VelocityAndTracer &operator=(const VelocityAndTracer &) = default;
  VelocityAndTracer &operator=(VelocityAndTracer &) = default;

  virtual void initialize(const zisa::array_view<real_t, Dim + 1> &u) override {
    zisa::shape_t<Dim + 1> shape_u = u.shape();
    shape_u[0] -= 1;
    zisa::array_view<real_t, Dim + 1> view_u(
        shape_u, u.raw(), u.memory_location());
    zisa::shape_t<Dim + 1> shape_rho = u.shape();
    shape_rho[0] = 1;
    zisa::array_view<real_t, Dim + 1> view_rho(
        shape_rho, u.raw() + zisa::product(shape_u), u.memory_location());
    init_u_->initialize(view_u);
    init_rho_->initialize(view_rho);
  }

  virtual void
  initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat) override {
    zisa::shape_t<Dim + 1> shape_u_hat = u_hat.shape();
    shape_u_hat[0] -= 1;
    zisa::array_view<complex_t, Dim + 1> view_u_hat(
        shape_u_hat, u_hat.raw(), u_hat.memory_location());
    zisa::shape_t<Dim + 1> shape_rho = u_hat.shape();
    shape_rho[0] = 1;
    zisa::array_view<complex_t, Dim + 1> view_rho(
        shape_rho,
        u_hat.raw() + zisa::product(shape_u_hat),
        u_hat.memory_location());
    init_u_->initialize(view_u_hat);
    init_rho_->initialize(view_rho);
  }

protected:
  virtual void
  do_initialize(const zisa::array_view<real_t, dim_v + 1> &) override {}
  virtual void
  do_initialize(const zisa::array_view<complex_t, dim_v + 1> &) override {}

private:
  std::shared_ptr<Initializer<Dim>> init_u_;
  std::shared_ptr<Initializer<Dim>> init_rho_;
};

}

#endif
