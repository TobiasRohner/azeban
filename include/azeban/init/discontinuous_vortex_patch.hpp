#ifndef DISCONTINUOUS_VORTEX_PATCH_H_
#define DISCONTINUOUS_VORTEX_PATCH_H_

#include "initializer.hpp"

namespace azeban {

class DiscontinuousVortexPatch final : public Initializer<2> {
  using super = Initializer<2>;

public:
  DiscontinuousVortexPatch() = default;
  DiscontinuousVortexPatch(const DiscontinuousVortexPatch &) = default;
  DiscontinuousVortexPatch(DiscontinuousVortexPatch &&) = default;

  virtual ~DiscontinuousVortexPatch() override = default;

  DiscontinuousVortexPatch &operator=(const DiscontinuousVortexPatch &)
      = default;
  DiscontinuousVortexPatch &operator=(DiscontinuousVortexPatch &) = default;

  virtual void initialize(const zisa::array_view<real_t, 3> &u) const override;
  virtual void
  initialize(const zisa::array_view<complex_t, 3> &u_hat) const override;
};

}

#endif
