/*
 * This file is part of azeban (https://github.com/TobiasRohner/azeban).
 * Copyright (c) 2021 Tobias Rohner.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef BROWNIAN_MOTION_H_
#define BROWNIAN_MOTION_H_

#include "initializer.hpp"
#include <azeban/random/normal.hpp>
#include <azeban/random/random_variable.hpp>
#include <azeban/random/uniform.hpp>

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
        uniform_(std::make_shared<Uniform<real_t, RNG>>(-1, 1, rng)) {}
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
  RandomVariable<real_t> uniform_;
};

template <>
class BrownianMotion<3> final : public Initializer<3> {
  using super = Initializer<3>;

public:
  template <typename RNG>
  BrownianMotion(const RandomVariable<real_t> &hurst, RNG &rng)
      : hurst_(hurst),
        uniform_(std::make_shared<Uniform<real_t, RNG>>(-1, 1, rng)) {}
  BrownianMotion(const BrownianMotion &) = default;
  BrownianMotion(BrownianMotion &&) = default;

  virtual ~BrownianMotion() override = default;

  BrownianMotion &operator=(const BrownianMotion &) = default;
  BrownianMotion &operator=(BrownianMotion &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 4> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 4> &u_hat) override;
#if AZEBAN_HAS_MPI
  virtual void do_initialize(const zisa::array_view<real_t, 4> &u,
                             const Grid<3> &grid,
                             const Communicator *comm,
                             void *work_area = nullptr) override;
  virtual void do_initialize(const zisa::array_view<complex_t, 4> &u_hat,
                             const Grid<3> &grid,
                             const Communicator *comm,
                             void *work_area = nullptr) override;
#endif

private:
  RandomVariable<real_t> hurst_;
  RandomVariable<real_t> uniform_;
};

}

#endif
