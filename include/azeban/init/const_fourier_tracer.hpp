#ifndef CONST_FOURIER_TRACER_HPP
#define CONST_FOURIER_TRACER_HPP

#include "initializer.hpp"
#include <array>
#include <azeban/operations/fft.hpp>
#include <type_traits>

namespace azeban {

template <int Dim>
class ConstFourierTracer final : public Initializer<Dim> {
  using super = Initializer<Dim>;

public:
  ConstFourierTracer(real_t rho) : rho_(rho) {}
  ConstFourierTracer(const ConstFourierTracer &) = default;
  ConstFourierTracer(ConstFourierTracer &&) = default;

  virtual ~ConstFourierTracer() = default;

  ConstFourierTracer &operator=(const ConstFourierTracer &) = default;
  ConstFourierTracer &operator=(ConstFourierTracer &&) = default;

  virtual void initialize(const zisa::array_view<real_t, Dim + 1> &u) override {
    const zisa::int_t N = u.shape(1);
    zisa::shape_t<Dim + 1> shape_fourier = u.shape();
    shape_fourier[Dim] = N / 2 + 1;
    zisa::array<real_t, Dim + 1> h_u(u.shape());
    zisa::array<complex_t, Dim + 1> h_u_hat(shape_fourier);
    const auto fft = make_fft<Dim>(h_u_hat, h_u);
    do_initialize(h_u_hat);
    fft->backward();
    for (auto &r : h_u) {
      r /= zisa::pow<Dim>(N);
    }
    zisa::copy(u, h_u);
  }

  virtual void
  initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat) override {
    const auto init = [&](const zisa::array_view<complex_t, Dim + 1> &u_) {
      for (auto &c : u_) {
        c = rho_;
      }
    };
    if (u_hat.memory_location() == zisa::device_type::cpu) {
      init(u_hat);
    } else if (u_hat.memory_location() == zisa::device_type::cuda) {
      zisa::array<complex_t, Dim + 1> h_u_hat(u_hat.shape());
      init(h_u_hat);
      zisa::copy(u_hat, h_u_hat);
    } else {
      LOG_ERR("Unsupported memory loaction");
    }
  }

protected:
  virtual void
  do_initialize(const zisa::array_view<real_t, Dim + 1> &u) override {}
  virtual void
  do_initialize(const zisa::array_view<complex_t, Dim + 1> &uhat) override {}

private:
  real_t rho_;
};

}

#endif
