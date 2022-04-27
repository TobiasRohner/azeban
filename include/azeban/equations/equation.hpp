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
#ifndef EQUATION_H_
#define EQUATION_H_

#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
class Equation {
public:
  static constexpr int dim_v = Dim;

  Equation(const Grid<Dim> &grid) : grid_(grid){};
  Equation(const Equation &) = default;
  Equation(Equation &&) = default;
  virtual ~Equation() = default;
  Equation &operator=(const Equation &) = default;
  Equation &operator=(Equation &&) = default;

  // Replaces the contents of u with its time derivative
  virtual void dudt(const zisa::array_view<complex_t, dim_v + 1> &dudt,
                    const zisa::array_const_view<complex_t, dim_v + 1> &u)
      = 0;

  virtual real_t dt() const = 0;

  virtual int n_vars() const = 0;
  virtual real_t visc() const { return 0; }

  virtual void *get_fft_work_area() const { return nullptr; }

protected:
  Grid<Dim> grid_;
};

}

#endif
