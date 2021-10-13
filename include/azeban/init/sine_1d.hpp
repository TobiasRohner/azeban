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
#ifndef SINE_1D_H_
#define SINE_1D_H_

#include "initializer.hpp"

namespace azeban {

class Sine1D final : public Initializer<1> {
  using super = Initializer<1>;

public:
  Sine1D() = default;
  Sine1D(const Sine1D &) = default;
  Sine1D(Sine1D &&) = default;

  virtual ~Sine1D() override = default;

  Sine1D &operator=(const Sine1D &) = default;
  Sine1D &operator=(Sine1D &&) = default;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 2> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 2> &u_hat) override;
};

}

#endif
