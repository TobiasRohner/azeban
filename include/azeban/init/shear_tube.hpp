#ifndef SHEAR_TUBE_H_
#define SHEAR_TUBE_H_

#include "initializer.hpp"

namespace azeban {

class ShearTube final : public Initializer<3> {
  using super = Initializer<3>;

public:
  ShearTube(real_t rho, real_t delta) : rho_(rho), delta_(delta) {}
  ShearTube(const ShearTube &) = default;
  ShearTube(ShearTube &&) = default;

  virtual ~ShearTube() override = default;

  ShearTube &operator=(const ShearTube &) = default;
  ShearTube &operator=(ShearTube &&) = default;

protected:
  virtual void
  do_initialize(const zisa::array_view<real_t, 4> &u) const override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 4> &u_hat) const override;

private:
  real_t rho_;
  real_t delta_;
};

}

#endif
