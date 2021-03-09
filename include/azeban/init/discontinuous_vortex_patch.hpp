#ifndef DISCONTINUOUS_VORTEX_PATCH_H_
#define DISCONTINUOUS_VORTEX_PATCH_H_

#include "initializer.hpp"
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>

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

  virtual void initialize(const zisa::array_view<real_t, 3> &u) const override {
    const auto init = [&](auto &&u_) {
      const zisa::int_t N = u_.shape(1);
      for (zisa::int_t i = 0; i < N; ++i) {
        for (zisa::int_t j = 0; j < N; ++j) {
          const real_t x = static_cast<real_t>(i) / N;
          const real_t y = static_cast<real_t>(j) / N;
          const real_t r
              = zisa::sqrt(zisa::pow<2>(x - 0.5) + zisa::pow<2>(y - 0.5));
          u_(0, i, j) = r < 0.25 ? -0.5 * (y - 0.5) : 0;
          u_(1, i, j) = r < 0.25 ? 0.5 * (x - 0.5) : 0;
        }
      }
    };
    if (u.memory_location() == zisa::device_type::cpu) {
      init(u);
    } else if (u.memory_location() == zisa::device_type::cuda) {
      auto h_u = zisa::array<real_t, 3>(u.shape(), zisa::device_type::cpu);
      init(h_u);
      zisa::copy(u, h_u);
    } else {
      LOG_ERR("Unknown Memory Location");
    }
  }

  virtual void
  initialize(const zisa::array_view<complex_t, 3> &u_hat) const override {
    const zisa::int_t N = u_hat.shape(1);
    auto u = zisa::array<real_t, 3>(zisa::shape_t<3>(2, N, N),
                                    u_hat.memory_location());
    auto fft = make_fft<2>(u_hat, u);
    initialize(u);
    fft->forward();
  }
};

}

#endif
