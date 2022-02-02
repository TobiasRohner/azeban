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

#ifndef TIME_INTEGRATOR_H_
#define TIME_INTEGRATOR_H_

#include <azeban/config.hpp>
#include <azeban/equations/equation.hpp>
#include <zisa/config.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
class TimeIntegrator {
public:
  static constexpr int dim_v = Dim;

  TimeIntegrator() = delete;
  TimeIntegrator(zisa::device_type device,
                 const std::shared_ptr<Equation<dim_v>> &equation)
      : device_(device), equation_(equation) {}
  TimeIntegrator(const TimeIntegrator &) = default;
  TimeIntegrator(TimeIntegrator &&) = default;

  virtual ~TimeIntegrator() = default;

  TimeIntegrator &operator=(const TimeIntegrator &) = default;
  TimeIntegrator &operator=(TimeIntegrator &&) = default;

  virtual void integrate(real_t dt,
                         const zisa::array_view<complex_t, dim_v + 1> &u)
      = 0;

  zisa::device_type memory_location() const { return device_; }
  std::shared_ptr<Equation<dim_v>> equation() { return equation_; }
  std::shared_ptr<const Equation<dim_v>> equation() const { return equation_; }

protected:
  zisa::device_type device_;
  std::shared_ptr<Equation<dim_v>> equation_;
};

}

#endif
