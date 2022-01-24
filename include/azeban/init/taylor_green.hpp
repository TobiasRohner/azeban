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
