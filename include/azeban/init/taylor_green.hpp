#ifndef TAYLOR_GREEN_H_
#define TAYLOR_GREEN_H_

#include "initializer.hpp"

namespace azeban {

template <int Dim>
class TaylorGreen final : public Initializer<Dim> {
  static_assert(Dim == 2 || Dim == 3,
                "Taylor Green is only implemented for 2D or 3D");
};

template <>
class TaylorGreen<2> final : public Initializer<2> {
  using super = Initializer<2>;

public:
  TaylorGreen() = default;
  TaylorGreen(const TaylorGreen &) = default;
  TaylorGreen(TaylorGreen &&) = default;

  virtual ~TaylorGreen() override = default;

  TaylorGreen &operator=(const TaylorGreen &) = default;
  TaylorGreen &operator=(TaylorGreen &&) = default;

  virtual void initialize(const zisa::array_view<real_t, 3> &u) const override;
  virtual void
  initialize(const zisa::array_view<complex_t, 3> &u_hat) const override;
};

template <>
class TaylorGreen<3> final : public Initializer<3> {
  using super = Initializer<3>;

public:
  TaylorGreen() = default;
  TaylorGreen(const TaylorGreen &) = default;
  TaylorGreen(TaylorGreen &&) = default;

  virtual ~TaylorGreen() override = default;

  TaylorGreen &operator=(const TaylorGreen &) = default;
  TaylorGreen &operator=(TaylorGreen &&) = default;

  virtual void initialize(const zisa::array_view<real_t, 4> &u) const override;
  virtual void
  initialize(const zisa::array_view<complex_t, 4> &u_hat) const override;
};

}

#endif
