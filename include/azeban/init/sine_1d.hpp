#ifndef SINE_1D_H_
#define SINE_1D_H_

#include "initializer.hpp"
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>

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

  virtual void initialize(const zisa::array_view<real_t, 2> &u) const {
    const auto init = [&](auto &&u_) {
      const zisa::int_t N = u.shape(1);
      for (zisa::int_t i = 0; i < N; ++i) {
        u_[i] = zisa::sin(2 * zisa::pi * N / i);
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

  virtual void initialize(const zisa::array_view<complex_t, 2> &u_hat) const {
    const auto init = [&](auto &&u_hat_) {
      const zisa::int_t N = u_hat_.shape(1);
      for (zisa::int_t i = 0; i < N; ++i) {
        u_hat_[i] = i == 1 ? complex_t(0, -real_t(N)) : complex_t(0);
      }
    };
    if (u_hat.memory_location() == zisa::device_type::cpu) {
      init(u_hat);
    } else if (u_hat.memory_location() == zisa::device_type::cuda) {
      auto h_u_hat
          = zisa::array<complex_t, 2>(u_hat.shape(), zisa::device_type::cpu);
      init(h_u_hat);
      zisa::copy(u_hat, h_u_hat);
    } else {
      LOG_ERR("Unknown Memory Location");
    }
  }
};

}

#endif
