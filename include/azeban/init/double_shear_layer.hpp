#ifndef DOUBLE_SHEAR_LAYER_H_
#define DOUBLE_SHEAR_LAYER_H_

#include "initializer.hpp"
#include <azeban/random/random_variable.hpp>

namespace azeban {

class DoubleShearLayer final : public Initializer<2> {
  using super = Initializer<2>;

public:
  DoubleShearLayer(const RandomVariable<real_t> &rho,
                   const RandomVariable<real_t> &delta)
      : rho_(rho), delta_(delta) {}
  DoubleShearLayer(const DoubleShearLayer &) = default;
  DoubleShearLayer(DoubleShearLayer &&) = default;

  virtual ~DoubleShearLayer() override = default;

  DoubleShearLayer &operator=(const DoubleShearLayer &) = default;
  DoubleShearLayer &operator=(DoubleShearLayer &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 3> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 3> &u_hat) override;

private:
  RandomVariable<real_t> rho_;
  RandomVariable<real_t> delta_;
};

}

#endif
