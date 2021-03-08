#ifndef SHOCK_H_
#define SHOCK_H_

#include "initializer.hpp"
#include <azeban/fft.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

class Shock final : public Initializer<1> {
  using super = Initializer<1>;

public:
  Shock(real_t x0, real_t x1) : x0_(x0), x1_(x1) {}
  Shock(const Shock &) = default;
  Shock(Shock &&) = default;

  virtual ~Shock() override = default;

  Shock &operator=(const Shock &) = default;
  Shock &operator=(Shock &&) = default;

  virtual void initialize(const zisa::array_view<real_t, 2> &u) const override {
    const auto init = [&](auto &&u_) {
      const zisa::int_t N = u_.shape(1);
      for (zisa::int_t i = 0; i < N; ++i) {
        const real_t x = static_cast<real_t>(i) / N;
        u_[i] = x >= x0_ && x < x1_ ? 1 : 0;
      }
    };
    if (u.memory_location() == zisa::device_type::cpu) {
      init(u);
    } else if (u.memory_location() == zisa::device_type::cuda) {
      auto h_u = zisa::array<real_t, 2>(u.shape(), zisa::device_type::cpu);
      init(h_u);
      zisa::copy(u, h_u);
    } else {
      LOG_ERR("Unknown Memory Location");
    }
  }

  virtual void
  initialize(const zisa::array_view<complex_t, 2> &u_hat) const override {
    const zisa::int_t N = 2 * (u_hat.shape(1) - 1);
    auto u = zisa::array<real_t, 2>(zisa::shape_t<2>(1, N),
                                    u_hat.memory_location());
    auto fft = make_fft<1>(u_hat, u);
    initialize(u);
    fft->forward();
  }

private:
  real_t x0_;
  real_t x1_;
};

}

#endif
