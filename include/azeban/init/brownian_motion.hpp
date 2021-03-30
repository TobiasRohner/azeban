#ifndef BROWNIAN_MOTION_H_
#define BROWNIAN_MOTION_H_

#include "initializer.hpp"
#include <azeban/random/normal.hpp>
#include <azeban/random/random_variable.hpp>

namespace azeban {

template <int Dim>
class BrownianMotion final : public Initializer<Dim> {};

template <>
class BrownianMotion<1> final : public Initializer<1> {
  using super = Initializer<1>;

public:
  template <typename RNG>
  BrownianMotion(const RandomVariable<real_t> &hurst, RNG &rng)
      : hurst_(hurst),
        normal_(std::make_shared<Normal<real_t, RNG>>(0, 1, rng)) {}
  BrownianMotion(const BrownianMotion &) = default;
  BrownianMotion(BrownianMotion &&) = default;

  virtual ~BrownianMotion() override = default;

  BrownianMotion &operator=(const BrownianMotion &) = default;
  BrownianMotion &operator=(BrownianMotion &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 2> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 2> &u_hat) override;

private:
  RandomVariable<real_t> hurst_;
  RandomVariable<real_t> normal_;
  void generate_step(const zisa::array_view<real_t, 1> &u,
                     real_t H,
                     zisa::int_t i0,
                     zisa::int_t i1);
};

template <>
class BrownianMotion<2> final : public Initializer<2> {
  using super = Initializer<2>;

public:
  template <typename RNG>
  BrownianMotion(const RandomVariable<real_t> &hurst, RNG &rng)
      : hurst_(hurst),
        normal_(std::make_shared<Normal<real_t, RNG>>(0, 1, rng)) {}
  BrownianMotion(const BrownianMotion &) = default;
  BrownianMotion(BrownianMotion &&) = default;

  virtual ~BrownianMotion() override = default;

  BrownianMotion &operator=(const BrownianMotion &) = default;
  BrownianMotion &operator=(BrownianMotion &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 3> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 3> &u_hat) override;

private:
  RandomVariable<real_t> hurst_;
  RandomVariable<real_t> normal_;
  void generate_step(const zisa::array_view<real_t, 2> &u,
                     real_t H,
                     zisa::int_t i0,
                     zisa::int_t i1,
                     zisa::int_t j0,
                     zisa::int_t j1);
};

template <>
class BrownianMotion<3> final : public Initializer<3> {
  using super = Initializer<3>;

public:
  template <typename RNG>
  BrownianMotion(const RandomVariable<real_t> &hurst, RNG &rng)
      : hurst_(hurst),
        normal_(std::make_shared<Normal<real_t, RNG>>(0, 1, rng)) {}
  BrownianMotion(const BrownianMotion &) = default;
  BrownianMotion(BrownianMotion &&) = default;

  virtual ~BrownianMotion() override = default;

  BrownianMotion &operator=(const BrownianMotion &) = default;
  BrownianMotion &operator=(BrownianMotion &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 4> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 4> &u_hat) override;

private:
  RandomVariable<real_t> hurst_;
  RandomVariable<real_t> normal_;
  void generate_step(const zisa::array_view<real_t, 3> &u,
                     real_t H,
                     zisa::int_t i0,
                     zisa::int_t i1,
                     zisa::int_t j0,
                     zisa::int_t j1,
                     zisa::int_t k0,
                     zisa::int_t k1);
};

}

#endif
