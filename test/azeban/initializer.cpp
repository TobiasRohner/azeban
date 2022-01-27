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
#include <azeban/catch.hpp>

#include <azeban/init/brownian_motion.hpp>
#include <azeban/init/discontinuous_vortex_patch.hpp>
#include <azeban/init/double_shear_layer.hpp>
#include <azeban/init/init_3d_from_2d.hpp>
#include <azeban/init/shear_tube.hpp>
#include <azeban/init/taylor_green.hpp>
#include <azeban/init/taylor_vortex.hpp>
#include <azeban/operations/copy_to_padded.hpp>
#include <azeban/random/delta.hpp>
#include <iostream>
#include <vector>

template <typename T, int D>
static zisa::array_view<T, D - 1> component(const zisa::array_view<T, D> &arr,
                                            zisa::int_t n) {
  zisa::shape_t<D - 1> slice_shape;
  for (zisa::int_t i = 0; i < D - 1; ++i) {
    slice_shape[i] = arr.shape(i + 1);
  }
  return zisa::array_view<T, D - 1>(slice_shape,
                                    arr.raw() + n * zisa::product(slice_shape),
                                    arr.memory_location());
}

template <typename T, int D>
static zisa::array_const_view<T, D - 1>
component(const zisa::array_const_view<T, D> &arr, zisa::int_t n) {
  zisa::shape_t<D - 1> slice_shape;
  for (zisa::int_t i = 0; i < D - 1; ++i) {
    slice_shape[i] = arr.shape(i + 1);
  }
  return zisa::array_const_view<T, D - 1>(slice_shape,
                                          arr.raw()
                                              + n * zisa::product(slice_shape),
                                          arr.memory_location());
}

template <typename T, int D>
static zisa::array_view<T, D - 1> component(zisa::array<T, D> &arr,
                                            zisa::int_t n) {
  zisa::shape_t<D - 1> slice_shape;
  for (zisa::int_t i = 0; i < D - 1; ++i) {
    slice_shape[i] = arr.shape(i + 1);
  }
  return zisa::array_view<T, D - 1>(
      slice_shape, arr.raw() + n * zisa::product(slice_shape), arr.device());
}

template <int dim_v>
static azeban::real_t measureConvergence(
    const std::shared_ptr<azeban::Initializer<dim_v>> &initializer,
    zisa::int_t N_ref) {
  zisa::shape_t<dim_v + 1> shape_ref;
  shape_ref[0] = dim_v;
  for (zisa::int_t i = 1; i < dim_v; ++i) {
    shape_ref[i] = N_ref;
  }
  shape_ref[dim_v] = N_ref / 2 + 1;
  auto u_ref_hat = zisa::array<azeban::complex_t, dim_v + 1>(shape_ref);
  initializer->initialize(u_ref_hat);

  std::vector<zisa::int_t> Ns;
  std::vector<azeban::real_t> errs;
  for (zisa::int_t N = 16; N < N_ref; N <<= 1) {
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = dim_v;
    for (zisa::int_t i = 1; i < dim_v; ++i) {
      shape[i] = N;
    }
    shape[dim_v] = N / 2 + 1;
    auto u_hat = zisa::array<azeban::complex_t, dim_v + 1>(shape);
    initializer->initialize(u_hat);
    auto u_pad_hat = zisa::array<azeban::complex_t, dim_v + 1>(shape_ref);
    for (zisa::int_t i = 0; i < dim_v; ++i) {
      azeban::copy_to_padded(component(u_pad_hat, i), component(u_hat, i), 0);
    }
    for (zisa::int_t i = 0; i < u_pad_hat.size(); ++i) {
      u_pad_hat[i] *= zisa::pow<dim_v>(static_cast<azeban::real_t>(N_ref) / N);
    }
    azeban::real_t errL2 = 0;
    for (zisa::int_t i = 0; i < u_ref_hat.size(); ++i) {
      errL2 += azeban::abs2(u_pad_hat[i] - u_ref_hat[i]);
    }
    errL2 = zisa::sqrt(errL2) / zisa::pow<dim_v>(N_ref);
    Ns.push_back(N);
    errs.push_back(errL2);
  }

  std::cout << "L2 errors = [" << errs[0];
  for (zisa::int_t i = 1; i < errs.size(); ++i) {
    std::cout << ", " << errs[i];
  }
  std::cout << "]" << std::endl;

  const azeban::real_t conv_rate
      = (zisa::log(errs[0]) - zisa::log(errs[errs.size() - 1]))
        / zisa::log(Ns[Ns.size() - 1] / Ns[0]);
  std::cout << "Estimated convergence rate: " << conv_rate << std::endl;

  return conv_rate;
}

TEST_CASE("Convergence Double Shear Layer 2D", "[slow][initializer]") {
  const auto initializer = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const azeban::real_t conv_rate = measureConvergence<2>(initializer, 1024);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Double Shear Layer 3D const. x", "[slow][initializer]") {
  const auto initializer2d = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(0, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Double Shear Layer 3D const. y", "[slow][initializer]") {
  const auto initializer2d = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(1, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Double Shear Layer 3D const. z", "[slow][initializer]") {
  const auto initializer2d = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(2, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Shear Tube 2D", "[slow][initializer]") {
  const auto initializer = std::make_shared<azeban::ShearTube>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Taylor Vortex 2D", "[slow][initializer]") {
  const auto initializer = std::make_shared<azeban::TaylorVortex>();
  const azeban::real_t conv_rate = measureConvergence<2>(initializer, 1024);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Taylor Vortex 3D const. x", "[slow][initializer]") {
  const auto initializer2d = std::make_shared<azeban::TaylorVortex>();
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(0, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Taylor Vortex 3D const. y", "[slow][initializer]") {
  const auto initializer2d = std::make_shared<azeban::TaylorVortex>();
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(1, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Taylor Vortex 3D const. z", "[slow][initializer]") {
  const auto initializer2d = std::make_shared<azeban::TaylorVortex>();
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(2, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}
