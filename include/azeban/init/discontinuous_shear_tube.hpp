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
#ifndef DISCONTINUOUS_SHEAR_TUBE_H_
#define DISCONTINUOUS_SHEAR_TUBE_H_

#include "initializer.hpp"
#include <azeban/random/random_variable.hpp>

namespace azeban {

class DiscontinuousShearTube final : public Initializer<3> {
  using super = Initializer<3>;

public:
  DiscontinuousShearTube(zisa::int_t N,
                         const RandomVariable<real_t> &rho,
                         const RandomVariable<real_t> &delta,
                         const RandomVariable<real_t> &uniform)
      : N_(N), rho_(rho), delta_(delta), uniform_(uniform) {}
  DiscontinuousShearTube(const DiscontinuousShearTube &) = default;
  DiscontinuousShearTube(DiscontinuousShearTube &&) = default;

  virtual ~DiscontinuousShearTube() override = default;

  DiscontinuousShearTube &operator=(const DiscontinuousShearTube &) = default;
  DiscontinuousShearTube &operator=(DiscontinuousShearTube &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 4> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 4> &u_hat) override;

private:
  zisa::int_t N_;
  RandomVariable<real_t> rho_;
  RandomVariable<real_t> delta_;
  RandomVariable<real_t> uniform_;
};

}

#endif
