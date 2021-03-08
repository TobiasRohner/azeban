#ifndef INITIALIZER_H_
#define INITIALIZER_H_

#include <azeban/config.hpp>
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

  virtual void initialize(const zisa::array_view<real_t, Dim + 1> &u) const = 0;
  virtual void
  initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat) const = 0;
};

}

#endif
