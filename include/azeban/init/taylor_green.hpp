#ifndef TAYLOR_GREEN_H_
#define TAYLOR_GREEN_H_

#include "initializer.hpp"
#include <azeban/random/delta.hpp>
#include <azeban/random/random_variable.hpp>

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
  TaylorGreen() : delta_(std::make_shared<Delta<real_t>>(0)) {}
  TaylorGreen(const RandomVariable<real_t> &delta) : delta_(delta) {}
  TaylorGreen(const TaylorGreen &) = default;
  TaylorGreen(TaylorGreen &&) = default;

  virtual ~TaylorGreen() override = default;

  TaylorGreen &operator=(const TaylorGreen &) = default;
  TaylorGreen &operator=(TaylorGreen &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 3> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 3> &u_hat) override;

private:
  RandomVariable<real_t> delta_;
};

template <>
class TaylorGreen<3> final : public Initializer<3> {
  using super = Initializer<3>;

public:
  TaylorGreen() : delta_(std::make_shared<Delta<real_t>>(0)) {}
  TaylorGreen(const RandomVariable<real_t> &delta) : delta_(delta) {}
  TaylorGreen(const TaylorGreen &) = default;
  TaylorGreen(TaylorGreen &&) = default;

  virtual ~TaylorGreen() override = default;

  TaylorGreen &operator=(const TaylorGreen &) = default;
  TaylorGreen &operator=(TaylorGreen &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 4> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 4> &u_hat) override;

private:
  RandomVariable<real_t> delta_;
};

}

#endif
