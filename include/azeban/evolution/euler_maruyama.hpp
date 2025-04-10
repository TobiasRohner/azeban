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
#ifndef EULER_MARUYAMA_H_
#define EULER_MARUYAMA_H_

#include "time_integrator.hpp"
#include <azeban/operations/axpy.hpp>
#include <azeban/operations/clamp.hpp>
#include <azeban/operations/norm.hpp>
#include <azeban/operations/scale.hpp>
#include <azeban/profiler.hpp>

namespace azeban {

template <int Dim>
class EulerMaruyama final : public TimeIntegrator<Dim> {
  using super = TimeIntegrator<Dim>;

public:
  static constexpr int dim_v = Dim;

  EulerMaruyama() = delete;
  EulerMaruyama(zisa::device_type device,
                const zisa::shape_t<dim_v + 1> &shape,
                const std::shared_ptr<Equation<dim_v>> &equation)
      : super(device, equation), dudt_(shape, device) {}
  EulerMaruyama(const EulerMaruyama &) = delete;
  EulerMaruyama(EulerMaruyama &&) = default;

  virtual ~EulerMaruyama() override = default;

  EulerMaruyama &operator=(const EulerMaruyama &) = delete;
  EulerMaruyama &operator=(EulerMaruyama &&) = default;

  virtual double
  integrate(double t,
            double max_dt,
            double C,
            const zisa::array_view<complex_t, dim_v + 1> &u) override {
    ProfileHost profile("euler_maruyama::integrate");
    equation_->dudt(dudt_, u, t, max_dt, C);
    const double dt = zisa::min(equation_->dt(C), max_dt);
    const real_t max_allowed_step = zisa::pow<2 * dim_v>(u.shape(1)) / dt;
    const real_t current_step = norm(dudt_, 2);
    if (current_step > max_allowed_step) {
      scale(complex_t(max_allowed_step / current_step), dudt_.view());
    }
    // clamp(dudt_.view(), max_allowed_step);
    axpy(complex_t(dt), zisa::array_const_view<complex_t, dim_v + 1>(dudt_), u);
    return dt;
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
