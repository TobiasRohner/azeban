#ifndef SINE_1D_H_
#define SINE_1D_H_

#include "initializer.hpp"

namespace azeban {

class Sine1D final : public Initializer<1> {
  using super = Initializer<1>;

public:
  Sine1D() = default;
  Sine1D(const Sine1D &) = default;
  Sine1D(Sine1D &&) = default;

  virtual ~Sine1D() override = default;

  Sine1D &operator=(const Sine1D &) = default;
  Sine1D &operator=(Sine1D &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 2> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 2> &u_hat) override;
};

}

#endif
