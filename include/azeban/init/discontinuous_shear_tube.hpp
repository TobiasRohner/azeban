#ifndef DISCONTINUOUS_SHEAR_TUBE_H_
#define DISCONTINUOUS_SHEAR_TUBE_H_

#include "initializer.hpp"
#include <azeban/random/random_variable.hpp>

namespace azeban {

class DiscontinuousShearTube final : public Initializer<3> {
  using super = Initializer<3>;

public:
  DiscontinuousShearTube(zisa::int_t N,
	    const RandomVariable<real_t> &rho,
            const RandomVariable<real_t> &delta,
	    const RandomVariable<real_t> &uniform)
      : N_(N), rho_(rho), delta_(delta), uniform_(uniform) {}
  DiscontinuousShearTube(const DiscontinuousShearTube &) = default;
  DiscontinuousShearTube(DiscontinuousShearTube &&) = default;

  virtual ~DiscontinuousShearTube() override = default;

  DiscontinuousShearTube &operator=(const DiscontinuousShearTube &) = default;
  DiscontinuousShearTube &operator=(DiscontinuousShearTube &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 4> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 4> &u_hat) override;

private:
  zisa::int_t N_;
  RandomVariable<real_t> rho_;
  RandomVariable<real_t> delta_;
  RandomVariable<real_t> uniform_;
};

}

#endif
