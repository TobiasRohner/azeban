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
#ifndef SSP_RK2_H_
#define SSP_RK2_H_

#include "time_integrator.hpp"
#include <azeban/operations/axpy.hpp>
#include <azeban/profiler.hpp>

namespace azeban {

template <int Dim>
class SSP_RK2 final : public TimeIntegrator<Dim> {
  using super = TimeIntegrator<Dim>;

public:
  static constexpr int dim_v = Dim;

  SSP_RK2() = delete;
  SSP_RK2(zisa::device_type device,
          const zisa::shape_t<dim_v + 1> &shape,
          const std::shared_ptr<Equation<dim_v>> &equation)
      : super(device, equation), u_star_(shape, device), dudt_(shape, device) {}
  SSP_RK2(const SSP_RK2 &) = delete;
  SSP_RK2(SSP_RK2 &&) = default;

  virtual ~SSP_RK2() override = default;

  SSP_RK2 &operator=(const SSP_RK2 &) = delete;
  SSP_RK2 &operator=(SSP_RK2 &&) = default;

  virtual real_t
  integrate(real_t t,
            real_t max_dt,
            real_t C,
            const zisa::array_view<complex_t, dim_v + 1> &u) override {
    ProfileHost profile("SSP_RK2::integrate");
    zisa::copy(u_star_, u);
    equation_->dudt(dudt_, u, t, max_dt);
    const real_t dt = zisa::min(C * equation_->dt(), max_dt);
    axpy(complex_t(0.5 * dt),
         zisa::array_const_view<complex_t, dim_v + 1>(dudt_),
         u);
    axpy(complex_t(dt),
         zisa::array_const_view<complex_t, dim_v + 1>(dudt_),
         zisa::array_view<complex_t, dim_v + 1>(u_star_));
    equation_->dudt(u_star_, u_star_, t, dt);
    axpy(complex_t(0.5 * dt),
         zisa::array_const_view<complex_t, dim_v + 1>(u_star_),
         u);
    return dt;
  }

  using super::equation;
  using super::memory_location;

protected:
  using super::device_;
  using super::equation_;

private:
  zisa::array<complex_t, dim_v + 1> u_star_;
  zisa::array<complex_t, dim_v + 1> dudt_;
};

}

#endif
