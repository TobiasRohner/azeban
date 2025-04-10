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
#ifndef SSP_RK3_H_
#define SSP_RK3_H_

#include "time_integrator.hpp"
#include <azeban/operations/axpby.hpp>
#include <azeban/operations/axpy.hpp>
#include <azeban/profiler.hpp>

namespace azeban {

template <int Dim>
class SSP_RK3 : public TimeIntegrator<Dim> {
  using super = TimeIntegrator<Dim>;

public:
  static constexpr int dim_v = Dim;

  SSP_RK3() = delete;
  SSP_RK3(zisa::device_type device,
          const zisa::shape_t<dim_v + 1> &shape,
          const std::shared_ptr<Equation<dim_v>> &equation)
      : super(device, equation), u1_(shape, device), u2_(shape, device) {}
  SSP_RK3(const SSP_RK3 &) = delete;
  SSP_RK3(SSP_RK3 &&) = default;

  virtual ~SSP_RK3() override = default;

  SSP_RK3 &operator=(const SSP_RK3 &) = delete;
  SSP_RK3 &operator=(SSP_RK3 &&) = default;

  virtual double
  integrate(double t,
            double max_dt,
            double C,
            const zisa::array_view<complex_t, dim_v + 1> &u) override {
    ProfileHost profile("SSP_RK3::integrate");
    equation_->dudt(u1_, u, t, max_dt, C);
    const double dt = zisa::min(equation_->dt(C), max_dt);
    axpby<complex_t, dim_v + 1>(1, u, dt, u1_);
    equation_->dudt(u2_, u1_, t, dt, C);
    axpby<complex_t, dim_v + 1>(1, u1_, dt, u2_);
    axpby<complex_t, dim_v + 1>(3. / 4, u, 1. / 4, u2_);
    equation_->dudt(u1_, u2_, t, dt, C);
    axpby<complex_t, dim_v + 1>(1, u2_, dt, u1_);
    axpby<complex_t, dim_v + 1>(2. / 3, u1_, 1. / 3, u);
    return dt;
  }

  using super::equation;
  using super::memory_location;

protected:
  using super::device_;
  using super::equation_;

private:
  zisa::array<complex_t, dim_v + 1> u1_;
  zisa::array<complex_t, dim_v + 1> u2_;
};

}

#endif
