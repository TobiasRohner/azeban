#ifndef SHOCK_H_
#define SHOCK_H_

#include "initializer.hpp"
#include <azeban/random/random_variable.hpp>

namespace azeban {

class Shock final : public Initializer<1> {
  using super = Initializer<1>;

public:
  Shock(const RandomVariable<real_t> &x0, const RandomVariable<real_t> &x1)
      : x0_(x0), x1_(x1) {}
  Shock(const Shock &) = default;
  Shock(Shock &&) = default;

  virtual ~Shock() override = default;

  Shock &operator=(const Shock &) = default;
  Shock &operator=(Shock &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 2> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 2> &u_hat) override;

private:
  RandomVariable<real_t> x0_;
  RandomVariable<real_t> x1_;
};

}

#endif
