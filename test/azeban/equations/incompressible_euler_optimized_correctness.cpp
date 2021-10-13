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

#include <azeban/equations/incompressible_euler.hpp>
#include <azeban/equations/incompressible_euler_naive.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/evolution/ssp_rk3.hpp>
#include <azeban/init/discontinuous_vortex_patch.hpp>
#include <azeban/init/double_shear_layer.hpp>
#include <azeban/init/init_3d_from_2d.hpp>
#include <azeban/init/taylor_vortex.hpp>
#include <azeban/operations/fft.hpp>
#include <azeban/operations/operations.hpp>
#include <azeban/random/delta.hpp>
#include <azeban/simulation.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <vector>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>
#include <zisa/memory/array.hpp>

template <int dim_v>
static azeban::real_t measureConvergence(
    const std::shared_ptr<azeban::Initializer<dim_v>> &initializer,
    zisa::int_t N_ref,
    azeban::real_t t) {
  const auto solve_euler
      = [&](const zisa::array_view<azeban::real_t, dim_v + 1> &u,
            const azeban::Grid<dim_v> &grid,
            const std::shared_ptr<azeban::Equation<dim_v>> &equation) {
          const auto timestepper = std::make_shared<azeban::SSP_RK3<dim_v>>(
              zisa::device_type::cuda,
              grid.shape_fourier(equation->n_vars()),
              equation);
          azeban::CFL<dim_v> cfl(grid, 0.2);
          azeban::Simulation<dim_v> simulation(
              grid.shape_fourier(equation->n_vars()),
              cfl,
              timestepper,
              zisa::device_type::cuda);

          auto d_u = zisa::cuda_array<azeban::real_t, dim_v + 1>(
              grid.shape_phys(dim_v));
          const auto fft = azeban::make_fft<dim_v>(simulation.u(), d_u);

          initializer->initialize(simulation.u());
          simulation.simulate_until(t);
          fft->backward();
          zisa::copy(u, d_u);
          for (zisa::int_t i = 0; i < zisa::product(u.shape()); ++i) {
            u[i] /= zisa::product(u.shape()) / u.shape(0);
          }
        };

  azeban::Grid<dim_v> grid_ref(N_ref);
  azeban::SmoothCutoff1D visc_ref(0.05 / N_ref, 1);
  const auto equation_ref = std::make_shared<
      azeban::IncompressibleEulerNaive<dim_v, azeban::SmoothCutoff1D>>(
      grid_ref, visc_ref, zisa::device_type::cuda);
  auto u_ref
      = zisa::array<azeban::real_t, dim_v + 1>(grid_ref.shape_phys(dim_v));
  fmt::print(
      stderr, "Solving reference solution in {}d, N_ref = {}\n", dim_v, N_ref);
  solve_euler(u_ref, grid_ref, equation_ref);

  std::vector<zisa::int_t> Ns;
  std::vector<azeban::real_t> errs;
  for (zisa::int_t N = 16; N < N_ref; N <<= 1) {
    azeban::Grid<dim_v> grid(N);
    azeban::SmoothCutoff1D visc(0.05 / N, 1);
    const auto equation = std::make_shared<
        azeban::IncompressibleEuler<dim_v, azeban::SmoothCutoff1D>>(
        grid, visc, zisa::device_type::cuda);
    auto u = zisa::array<azeban::real_t, dim_v + 1>(grid.shape_phys(dim_v));
    fmt::print(stderr, "Solving {}d Problem, N = {}\n", dim_v, N);
    solve_euler(u, grid, equation);
    azeban::real_t errL2 = 0;
    if constexpr (dim_v == 2) {
      for (zisa::int_t i = 0; i < N; ++i) {
        for (zisa::int_t j = 0; j < N; ++j) {
          const zisa::int_t i_ref = i * N_ref / N;
          const zisa::int_t j_ref = j * N_ref / N;
          const azeban::real_t du = u(0, i, j) - u_ref(0, i_ref, j_ref);
          const azeban::real_t dv = u(1, i, j) - u_ref(1, i_ref, j_ref);
          errL2 += zisa::pow<2>(du) + zisa::pow<2>(dv);
        }
      }
      errL2 = zisa::sqrt(errL2) / (N * N);
    } else {
      for (zisa::int_t i = 0; i < N; ++i) {
        for (zisa::int_t j = 0; j < N; ++j) {
          for (zisa::int_t k = 0; k < N; ++k) {
            const zisa::int_t i_ref = i * N_ref / N;
            const zisa::int_t j_ref = j * N_ref / N;
            const zisa::int_t k_ref = k * N_ref / N;
            const azeban::real_t du
                = u(0, i, j, k) - u_ref(0, i_ref, j_ref, k_ref);
            const azeban::real_t dv
                = u(1, i, j, k) - u_ref(1, i_ref, j_ref, k_ref);
            const azeban::real_t dw
                = u(2, i, j, k) - u_ref(2, i_ref, j_ref, k_ref);
            errL2 += zisa::pow<2>(du) + zisa::pow<2>(dv) + zisa::pow<2>(dw);
          }
        }
      }
      errL2 = zisa::sqrt(errL2) / (N * N * N);
    }
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

TEST_CASE("Double Shear Layer 2D Optimized Correctness", "[slow],[optimized]") {
  const auto initializer = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const azeban::real_t conv_rate = measureConvergence<2>(initializer, 512, 1);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Double Shear Layer 3D const. x Optimized Correctness",
          "[slow],[optimized]") {
  const auto initializer2d = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(0, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 1);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Double Shear Layer 3D const. y Optimized Correctness",
          "[slow],[optimized]") {
  const auto initializer2d = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(1, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 1);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Double Shear Layer 3D const. z Optimized Correctness",
          "[slow],[optimized]") {
  const auto initializer2d = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(2, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 1);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Discontinous Vortex Patch 2D Optimized Correctness",
          "[slow],[optimized]") {
  const auto initializer = std::make_shared<azeban::DiscontinuousVortexPatch>();
  const azeban::real_t conv_rate = measureConvergence<2>(initializer, 512, 5);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Discontinous Vortex Patch 3D const x Optimized Correctness",
          "[slow],[optimized]") {
  const auto initializer2d
      = std::make_shared<azeban::DiscontinuousVortexPatch>();
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(0, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 5);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Discontinous Vortex Patch 3D const y Optimized Correctness",
          "[slow],[optimized]") {
  const auto initializer2d
      = std::make_shared<azeban::DiscontinuousVortexPatch>();
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(1, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 5);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Discontinous Vortex Patch 3D const z Optimized Correctness",
          "[slow],[optimized]") {
  const auto initializer2d
      = std::make_shared<azeban::DiscontinuousVortexPatch>();
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(2, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 5);
  REQUIRE(conv_rate >= 1);
}
