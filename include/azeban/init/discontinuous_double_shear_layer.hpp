#ifndef DISCONTINUOUS_DOUBLE_SHEAR_LAYER_H_
#define DISCONTINUOUS_DOUBLE_SHEAR_LAYER_H_

#include "initializer.hpp"
#include <azeban/random/random_variable.hpp>

namespace azeban {

class DiscontinuousDoubleShearLayer final : public Initializer<2> {
  using super = Initializer<2>;

public:
  DiscontinuousDoubleShearLayer(zisa::int_t N,
                                const RandomVariable<real_t> &rho,
                                const RandomVariable<real_t> &delta,
                                const RandomVariable<real_t> &uniform)
      : N_(N), rho_(rho), delta_(delta), uniform_(uniform) {}
  DiscontinuousDoubleShearLayer(const DiscontinuousDoubleShearLayer &)
      = default;
  DiscontinuousDoubleShearLayer(DiscontinuousDoubleShearLayer &&) = default;

  virtual ~DiscontinuousDoubleShearLayer() override = default;

  DiscontinuousDoubleShearLayer &
  operator=(const DiscontinuousDoubleShearLayer &)
      = default;
  DiscontinuousDoubleShearLayer &operator=(DiscontinuousDoubleShearLayer &&)
      = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 3> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 3> &u_hat) override;

private:
  zisa::int_t N_;
  RandomVariable<real_t> rho_;
  RandomVariable<real_t> delta_;
  RandomVariable<real_t> uniform_;
};

}

#endif
