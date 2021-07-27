#ifndef DISCONTINUOUS_DOUBLE_SHEAR_LAYER_H_
#define DISCONTINUOUS_DOUBLE_SHEAR_LAYER_H_

#include "initializer.hpp"
#include <azeban/random/random_variable.hpp>

namespace azeban {

class DiscontinuousDoubleShearLayer final : public Initializer<2> {
  using super = Initializer<2>;

public:
  DiscontinuousDoubleShearLayer(const RandomVariable<real_t> &delta)
      : delta_(delta) {}

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 3> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 3> &u_hat) override;

private:
  RandomVariable<real_t> delta_;
};

}

#endif
