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

#include "utils.hpp"
#include <azeban/grid.hpp>
#include <azeban/operations/fft.hpp>
#include <cmath>
#include <fmt/core.h>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>

TEST_CASE("L2 Normal Distribution 2D", "[utils]") {
  const zisa::int_t N_ref = 1024;
  const azeban::Grid<2> grid_ref(N_ref);
  zisa::array<azeban::complex_t, 3> u_ref_hat(grid_ref.shape_fourier(1));
  zisa::fill(u_ref_hat, azeban::complex_t(0));
  for (zisa::int_t N = 32; N < N_ref; N <<= 1) {
    for (azeban::real_t xbar : {-0.25, 0., 0.25}) {
      for (azeban::real_t ybar : {-0.25, 0., 0.25}) {
        for (azeban::real_t sigmax : {0.05, 0.1, 0.15}) {
          for (azeban::real_t sigmay : {0.05, 0.1, 0.15}) {
            azeban::Grid<2> grid(N);
            zisa::array<azeban::real_t, 3> u(grid.shape_phys(1));
            zisa::array<azeban::complex_t, 3> u_hat(grid.shape_fourier(1));
            const auto fft = azeban::make_fft<2>(u_hat, u);
            for (zisa::int_t i = 0; i < N; ++i) {
              for (zisa::int_t j = 0; j < N; ++j) {
                const azeban::real_t x
                    = static_cast<azeban::real_t>(i) / N - 0.5;
                const azeban::real_t y
                    = static_cast<azeban::real_t>(j) / N - 0.5;
                u(0, i, j) = 1;
                u(0, i, j) *= zisa::exp(-zisa::pow<2>((x - xbar) / sigmax));
                u(0, i, j) *= zisa::exp(-zisa::pow<2>((y - ybar) / sigmay));
              }
            }
            fft->forward();
            const azeban::real_t norm = L2<2>(u_hat, u_ref_hat);
            azeban::real_t exact = zisa::pi * sigmax * sigmay / 8;
            exact *= std::erf(zisa::sqrt(2) / sigmax * (0.5 - xbar))
                     - std::erf(zisa::sqrt(2) / sigmax * (-0.5 - xbar));
            exact *= std::erf(zisa::sqrt(2) / sigmay * (0.5 - ybar))
                     - std::erf(zisa::sqrt(2) / sigmay * (-0.5 - ybar));
            exact = zisa::sqrt(exact);
            REQUIRE(std::fabs(norm - exact) <= 1e-4);
          }
        }
      }
    }
  }
}

TEST_CASE("L2 Normal Distribution 3D", "[utils]") {
  const zisa::int_t N_ref = 512;
  const azeban::Grid<3> grid_ref(N_ref);
  zisa::array<azeban::complex_t, 4> u_ref_hat(grid_ref.shape_fourier(1));
  zisa::fill(u_ref_hat, azeban::complex_t(0));
  for (zisa::int_t N = 32; N < N_ref; N <<= 1) {
    for (azeban::real_t xbar : {-0.25, 0., 0.25}) {
      for (azeban::real_t ybar : {-0.25, 0., 0.25}) {
        for (azeban::real_t zbar : {-0.25, 0., 0.25}) {
          for (azeban::real_t sigmax : {0.05, 0.1, 0.15}) {
            for (azeban::real_t sigmay : {0.05, 0.1, 0.15}) {
              for (azeban::real_t sigmaz : {0.05, 0.1, 0.15}) {
                azeban::Grid<3> grid(N);
                zisa::array<azeban::real_t, 4> u(grid.shape_phys(1));
                zisa::array<azeban::complex_t, 4> u_hat(grid.shape_fourier(1));
                const auto fft = azeban::make_fft<3>(u_hat, u);
                for (zisa::int_t i = 0; i < N; ++i) {
                  for (zisa::int_t j = 0; j < N; ++j) {
                    for (zisa::int_t k = 0; k < N; ++k) {
                      const azeban::real_t x
                          = static_cast<azeban::real_t>(i) / N - 0.5;
                      const azeban::real_t y
                          = static_cast<azeban::real_t>(j) / N - 0.5;
                      const azeban::real_t z
                          = static_cast<azeban::real_t>(k) / N - 0.5;
                      u(0, i, j, k) = 1;
                      u(0, i, j, k)
                          *= zisa::exp(-zisa::pow<2>((x - xbar) / sigmax));
                      u(0, i, j, k)
                          *= zisa::exp(-zisa::pow<2>((y - ybar) / sigmay));
                      u(0, i, j, k)
                          *= zisa::exp(-zisa::pow<2>((z - zbar) / sigmaz));
                    }
                  }
                }
                fft->forward();
                const azeban::real_t norm = L2<3>(u_hat, u_ref_hat);
                azeban::real_t exact = zisa::sqrt(zisa::pow<3>(zisa::pi) / 2)
                                       * sigmax * sigmay * sigmaz / 16;
                exact *= std::erf(zisa::sqrt(2) / sigmax * (0.5 - xbar))
                         - std::erf(zisa::sqrt(2) / sigmax * (-0.5 - xbar));
                exact *= std::erf(zisa::sqrt(2) / sigmay * (0.5 - ybar))
                         - std::erf(zisa::sqrt(2) / sigmay * (-0.5 - ybar));
                exact *= std::erf(zisa::sqrt(2) / sigmaz * (0.5 - zbar))
                         - std::erf(zisa::sqrt(2) / sigmaz * (-0.5 - zbar));
                exact = zisa::sqrt(exact);
                REQUIRE(std::fabs(norm - exact) <= 1e-4);
              }
            }
          }
        }
      }
    }
  }
}
