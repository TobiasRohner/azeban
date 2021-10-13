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
#ifndef INIT_3D_FROM_2D_H_
#define INIT_3D_FROM_2D_H_

#include "initializer.hpp"

namespace azeban {

class Init3DFrom2D final : public Initializer<3> {
  using super = Initializer<3>;

public:
  Init3DFrom2D(zisa::int_t dim, const std::shared_ptr<Initializer<2>> &init2d)
      : dim_(dim), init2d_(init2d) {}
  Init3DFrom2D(const Init3DFrom2D &) = default;
  Init3DFrom2D(Init3DFrom2D &&) = default;

  virtual ~Init3DFrom2D() override = default;

  Init3DFrom2D &operator=(const Init3DFrom2D &) = default;
  Init3DFrom2D &operator=(Init3DFrom2D &&) = default;

  virtual void initialize(const zisa::array_view<real_t, 4> &u) override;
  virtual void initialize(const zisa::array_view<complex_t, 4> &u_hat) override;

protected:
  virtual void do_initialize(const zisa::array_view<real_t, 4> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, 4> &u_hat) override;

private:
  zisa::int_t dim_;
  std::shared_ptr<Initializer<2>> init2d_;
};

}

#endif
