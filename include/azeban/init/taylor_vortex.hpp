#ifndef TAYLOR_VORTEX_H_
#define TAYLOR_VORTEX_H_

#include "initializer.hpp"

namespace azeban {

class TaylorVortex final : public Initializer<2> {
  using super = Initializer<2>;

public:
  TaylorVortex() = default;
  TaylorVortex(const TaylorVortex &) = default;
  TaylorVortex(TaylorVortex &&) = default;

  virtual ~TaylorVortex() override = default;

  TaylorVortex &operator=(const TaylorVortex &) = default;
  TaylorVortex &operator=(TaylorVortex &) = default;

  virtual void initialize(const zisa::array_view<real_t, 3> &u) override;
  virtual void initialize(const zisa::array_view<complex_t, 3> &u_hat) override;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 3> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 3> &u_hat) override;
};

}

#endif
