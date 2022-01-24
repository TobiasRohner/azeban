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
#ifndef DISCONTINUOUS_DOUBLE_SHEAR_LAYER_H_
#define DISCONTINUOUS_DOUBLE_SHEAR_LAYER_H_

#include "initializer.hpp"
#include <azeban/random/random_variable.hpp>

namespace azeban {

class DiscontinuousDoubleShearLayer final : public Initializer<2> {
  using super = Initializer<2>;

public:
  DiscontinuousDoubleShearLayer(zisa::int_t N,
                                const RandomVariable<real_t> &rho,
                                const RandomVariable<real_t> &delta,
                                const RandomVariable<real_t> &uniform)
      : N_(N), rho_(rho), delta_(delta), uniform_(uniform) {}
  DiscontinuousDoubleShearLayer(const DiscontinuousDoubleShearLayer &)
      = default;
  DiscontinuousDoubleShearLayer(DiscontinuousDoubleShearLayer &&) = default;

  virtual ~DiscontinuousDoubleShearLayer() override = default;

  DiscontinuousDoubleShearLayer &
  operator=(const DiscontinuousDoubleShearLayer &)
      = default;
  DiscontinuousDoubleShearLayer &operator=(DiscontinuousDoubleShearLayer &&)
      = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 3> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 3> &u_hat) override;

private:
  zisa::int_t N_;
  RandomVariable<real_t> rho_;
  RandomVariable<real_t> delta_;
  RandomVariable<real_t> uniform_;
};

}

#endif
