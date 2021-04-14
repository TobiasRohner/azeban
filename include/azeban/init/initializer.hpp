#ifndef INITIALIZER_H_
#define INITIALIZER_H_

#include <azeban/config.hpp>
#include <azeban/operations/fft.hpp>
#include <azeban/operations/leray.hpp>
#include <azeban/operations/scale.hpp>
#include <fmt/core.h>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
class Initializer {
public:
  Initializer() = default;
  Initializer(const Initializer &) = default;
  Initializer(Initializer &&) = default;

  virtual ~Initializer() = default;

  Initializer &operator=(const Initializer &) = default;
  Initializer &operator=(Initializer &&) = default;

  virtual void initialize(const zisa::array_view<real_t, Dim + 1> &u) {
    if constexpr (Dim > 1) {
      zisa::shape_t<Dim + 1> u_hat_shape = u.shape();
      u_hat_shape[Dim] = u.shape(Dim) / 2 + 1;
      auto u_hat
          = zisa::array<complex_t, Dim + 1>(u_hat_shape, u.memory_location());

      auto fft = make_fft<Dim>(u_hat, u);

      do_initialize(u);
      fft->forward();
      leray(u_hat);
      scale(complex_t(u.shape(0)) / zisa::product(u.shape()),
            zisa::array_view<complex_t, Dim + 1>(u_hat));
      fft->backward();
    } else {
      do_initialize(u);
    }
  }

  virtual void initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat) {
    if constexpr (Dim > 1) {
      do_initialize(u_hat);
      leray(u_hat);
    } else {
      do_initialize(u_hat);
    }
  }

protected:
  virtual void do_initialize(const zisa::array_view<real_t, Dim + 1> &u) = 0;
  virtual void do_initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat)
      = 0;
};

}

#endif
