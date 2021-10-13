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
#ifndef SHOCK_H_
#define SHOCK_H_

#include "initializer.hpp"
#include <azeban/random/random_variable.hpp>

namespace azeban {

class Shock final : public Initializer<1> {
  using super = Initializer<1>;

public:
  Shock(const RandomVariable<real_t> &x0, const RandomVariable<real_t> &x1)
      : x0_(x0), x1_(x1) {}
  Shock(const Shock &) = default;
  Shock(Shock &&) = default;

  virtual ~Shock() override = default;

  Shock &operator=(const Shock &) = default;
  Shock &operator=(Shock &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 2> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 2> &u_hat) override;

private:
  RandomVariable<real_t> x0_;
  RandomVariable<real_t> x1_;
};

}

#endif
