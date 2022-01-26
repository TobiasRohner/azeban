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
#ifndef FORWARD_EULER_H_
#define FORWARD_EULER_H_

#include "time_integrator.hpp"
#include <azeban/operations/axpy.hpp>
#include <azeban/profiler.hpp>

namespace azeban {

template <int Dim>
class ForwardEuler final : public TimeIntegrator<Dim> {
  using super = TimeIntegrator<Dim>;

public:
  static constexpr int dim_v = Dim;

  ForwardEuler() = delete;
  ForwardEuler(zisa::device_type device,
               const zisa::shape_t<dim_v + 1> &shape,
               const std::shared_ptr<Equation<dim_v>> &equation)
      : super(device, equation), dudt_(shape, device) {}
  ForwardEuler(const ForwardEuler &) = delete;
  ForwardEuler(ForwardEuler &&) = default;

  virtual ~ForwardEuler() override = default;

  ForwardEuler &operator=(const ForwardEuler &) = delete;
  ForwardEuler &operator=(ForwardEuler &&) = default;

  virtual void
  integrate(real_t dt,
            const zisa::array_view<complex_t, dim_v + 1> &u) override {
    AZEBAN_PROFILE_START("forward_euler::integrate");
    equation_->dudt(dudt_, u);
    axpy(complex_t(dt), zisa::array_const_view<complex_t, dim_v + 1>(dudt_), u);
    AZEBAN_PROFILE_STOP("forward_euler::integrate");
  }

  using super::equation;
  using super::memory_location;

protected:
  using super::device_;
  using super::equation_;

private:
  zisa::array<complex_t, dim_v + 1> dudt_;
};

}

#endif
