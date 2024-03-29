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
#ifndef SHEAR_TUBE_H_
#define SHEAR_TUBE_H_

#include "initializer.hpp"
#include <azeban/random/random_variable.hpp>

namespace azeban {

class ShearTube final : public Initializer<3> {
  using super = Initializer<3>;

public:
  ShearTube(const RandomVariable<real_t> &rho,
            const RandomVariable<real_t> &delta)
      : rho_(rho), delta_(delta) {}
  ShearTube(const ShearTube &) = default;
  ShearTube(ShearTube &&) = default;

  virtual ~ShearTube() override = default;

  ShearTube &operator=(const ShearTube &) = default;
  ShearTube &operator=(ShearTube &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 4> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 4> &u_hat) override;

private:
  RandomVariable<real_t> rho_;
  RandomVariable<real_t> delta_;
};

}

#endif
