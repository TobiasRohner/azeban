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

#include <array>
#include <azeban/grid.hpp>
#include <azeban/init/double_shear_layer.hpp>
#include <azeban/init/shear_tube.hpp>
#include <azeban/operations/copy_padded.hpp>
#include <azeban/operations/fft.hpp>
#include <azeban/operations/leray.hpp>
#include <azeban/operations/operations.hpp>
#include <azeban/random/delta.hpp>
#include <azeban/random/random_variable.hpp>
#include <map>
#include <random>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>
#include <zisa/memory/array.hpp>

TEST_CASE("axpy", "[operations]") {
  zisa::int_t n = 128;
  zisa::shape_t<1> shape{n};
  auto h_x = zisa::array<azeban::real_t, 1>(shape);
  auto d_x = zisa::cuda_array<azeban::real_t, 1>(shape);
  auto h_y = zisa::array<azeban::real_t, 1>(shape);
  auto d_y = zisa::cuda_array<azeban::real_t, 1>(shape);

  for (zisa::int_t i = 0; i < n; ++i) {
    h_x[i] = 2 * i;
    h_y[i] = n - i;
  }

  zisa::copy(d_x, h_x);
  zisa::copy(d_y, h_y);
  azeban::axpy(azeban::real_t(0.5),
               zisa::array_const_view<azeban::real_t, 1>(d_x),
               zisa::array_view<azeban::real_t, 1>(d_y));
  zisa::copy(h_x, d_x);
  zisa::copy(h_y, d_y);

  for (zisa::int_t i = 0; i < n; ++i) {
    azeban::real_t expected_x = 2 * i;
    azeban::real_t expected_y = n;
    REQUIRE(std::fabs(h_x[i] - expected_x) <= 1e-10);
    REQUIRE(std::fabs(h_y[i] - expected_y) <= 1e-10);
  }
}

TEST_CASE("norm complex CUDA", "[operations]") {
  zisa::int_t n = 1000 * 1000;
  zisa::shape_t<1> shape{n};
  auto h_u = zisa::array<azeban::complex_t, 1>(shape);
  auto d_u = zisa::cuda_array<azeban::complex_t, 1>(shape);

  for (zisa::int_t i = 0; i < n; ++i) {
    h_u[i] = 1;
  }

  zisa::copy(d_u, h_u);
  const azeban::real_t d
      = azeban::norm(zisa::array_const_view<azeban::complex_t, 1>(d_u), 2);

  std::cout << "norm = " << d << std::endl;
  REQUIRE(std::fabs(d - 1000) <= 1e-10);
}

TEST_CASE("Leray 2D CPU", "[operations]") {
  zisa::int_t N = 32;
  zisa::shape_t<3> rshape{2, N, N};
  zisa::shape_t<3> cshape{2, N, N / 2 + 1};
  zisa::array<azeban::real_t, 3> u(rshape);
  zisa::array<azeban::complex_t, 3> uhat(cshape);
  auto fft = azeban::make_fft<2>(uhat, u);

  azeban::RandomVariable<azeban::real_t> rho(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.2));
  azeban::RandomVariable<azeban::real_t> delta(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.05));
  azeban::DoubleShearLayer init(rho, delta);
  init.initialize(u);
  fft->forward();
  leray(uhat);
  fft->backward();
  for (zisa::int_t i = 0; i < N; ++i) {
    for (zisa::int_t j = 0; j < N; ++j) {
      const zisa::int_t im = (i + N - 1) % N;
      const zisa::int_t ip = (i + 1) % N;
      const zisa::int_t jm = (j + N - 1) % N;
      const zisa::int_t jp = (j + 1) % N;
      const azeban::real_t dudx = (u(0, ip, j) - u(0, im, j)) / N;
      const azeban::real_t dvdy = (u(1, i, jp) - u(1, i, jm)) / N;
      const azeban::real_t div = dudx + dvdy;
      REQUIRE(std::fabs(div) < 1e-10);
    }
  }
}

TEST_CASE("Leray 3D CPU", "[operations]") {
  zisa::int_t N = 32;
  zisa::shape_t<4> rshape{3, N, N, N};
  zisa::shape_t<4> cshape{3, N, N, N / 2 + 1};
  zisa::array<azeban::real_t, 4> u(rshape);
  zisa::array<azeban::complex_t, 4> uhat(cshape);
  auto fft = azeban::make_fft<3>(uhat, u);

  azeban::RandomVariable<azeban::real_t> rho(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.2));
  azeban::RandomVariable<azeban::real_t> delta(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.05));
  azeban::ShearTube init(rho, delta);
  init.initialize(u);
  fft->forward();
  leray(uhat);
  fft->backward();
  for (zisa::int_t i = 0; i < N; ++i) {
    for (zisa::int_t j = 0; j < N; ++j) {
      for (zisa::int_t k = 0; k < N; ++k) {
        const zisa::int_t im = (i + N - 1) % N;
        const zisa::int_t ip = (i + 1) % N;
        const zisa::int_t jm = (j + N - 1) % N;
        const zisa::int_t jp = (j + 1) % N;
        const zisa::int_t km = (k + N - 1) % N;
        const zisa::int_t kp = (k + 1) % N;
        const azeban::real_t dudx = (u(0, ip, j, k) - u(0, im, j, k)) / (N * N);
        const azeban::real_t dvdy = (u(1, i, jp, k) - u(1, i, jm, k)) / (N * N);
        const azeban::real_t dwdz = (u(2, i, j, kp) - u(2, i, j, km)) / (N * N);
        const azeban::real_t div = dudx + dvdy + dwdz;
        REQUIRE(std::fabs(div) < 1e-10);
      }
    }
  }
}

TEST_CASE("Leray 2D CUDA", "[operations]") {
  zisa::int_t N = 32;
  zisa::shape_t<3> rshape{2, N, N};
  zisa::shape_t<3> cshape{2, N, N / 2 + 1};
  zisa::array<azeban::real_t, 3> hu(rshape, zisa::device_type::cpu);
  zisa::array<azeban::real_t, 3> du(rshape, zisa::device_type::cuda);
  zisa::array<azeban::complex_t, 3> uhat(cshape, zisa::device_type::cuda);
  auto fft = azeban::make_fft<2>(uhat, du);

  azeban::RandomVariable<azeban::real_t> rho(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.2));
  azeban::RandomVariable<azeban::real_t> delta(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.05));
  azeban::DoubleShearLayer init(rho, delta);
  init.initialize(du);
  fft->forward();
  leray(uhat);
  fft->backward();
  zisa::copy(hu, du);
  for (zisa::int_t i = 0; i < N; ++i) {
    for (zisa::int_t j = 0; j < N; ++j) {
      const zisa::int_t im = (i + N - 1) % N;
      const zisa::int_t ip = (i + 1) % N;
      const zisa::int_t jm = (j + N - 1) % N;
      const zisa::int_t jp = (j + 1) % N;
      const azeban::real_t dudx = (hu(0, ip, j) - hu(0, im, j)) / N;
      const azeban::real_t dvdy = (hu(1, i, jp) - hu(1, i, jm)) / N;
      const azeban::real_t div = dudx + dvdy;
      REQUIRE(std::fabs(div) < 1e-10);
    }
  }
}

TEST_CASE("Leray 3D CUDA", "[operations]") {
  zisa::int_t N = 32;
  zisa::shape_t<4> rshape{3, N, N, N};
  zisa::shape_t<4> cshape{3, N, N, N / 2 + 1};
  zisa::array<azeban::real_t, 4> hu(rshape, zisa::device_type::cpu);
  zisa::array<azeban::real_t, 4> du(rshape, zisa::device_type::cuda);
  zisa::array<azeban::complex_t, 4> uhat(cshape, zisa::device_type::cuda);
  auto fft = azeban::make_fft<3>(uhat, du);

  azeban::RandomVariable<azeban::real_t> rho(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.2));
  azeban::RandomVariable<azeban::real_t> delta(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.05));
  azeban::ShearTube init(rho, delta);
  init.initialize(du);
  fft->forward();
  leray(uhat);
  fft->backward();
  zisa::copy(hu, du);
  for (zisa::int_t i = 0; i < N; ++i) {
    for (zisa::int_t j = 0; j < N; ++j) {
      for (zisa::int_t k = 0; k < N; ++k) {
        const zisa::int_t im = (i + N - 1) % N;
        const zisa::int_t ip = (i + 1) % N;
        const zisa::int_t jm = (j + N - 1) % N;
        const zisa::int_t jp = (j + 1) % N;
        const zisa::int_t km = (k + N - 1) % N;
        const zisa::int_t kp = (k + 1) % N;
        const azeban::real_t dudx
            = (hu(0, ip, j, k) - hu(0, im, j, k)) / (N * N);
        const azeban::real_t dvdy
            = (hu(1, i, jp, k) - hu(1, i, jm, k)) / (N * N);
        const azeban::real_t dwdz
            = (hu(2, i, j, kp) - hu(2, i, j, km)) / (N * N);
        const azeban::real_t div = dudx + dvdy + dwdz;
        REQUIRE(std::fabs(div) < 1e-10);
      }
    }
  }
}

static long to_real_k(zisa::int_t k, zisa::int_t N) {
  long k_ = zisa::integer_cast<long>(k);
  if (k_ >= zisa::integer_cast<long>(N / 2 + 1)) {
    k_ -= N;
  }
  return k_;
}

static void
test_is_zero_padded(const zisa::array<azeban::complex_t, 1> &unpadded,
                    const zisa::array<azeban::complex_t, 1> &padded) {
  std::map<zisa::int_t, azeban::complex_t> m;
  for (zisa::int_t i = 0; i < unpadded.shape(0); ++i) {
    m[i] = unpadded(i);
  }
  for (zisa::int_t i = 0; i < padded.shape(0); ++i) {
    if (const auto it = m.find(i); it != m.end()) {
      REQUIRE(it->second == padded(i));
    } else {
      REQUIRE(padded(i) == 0);
    }
  }
}

static void
test_is_zero_padded(const zisa::array<azeban::complex_t, 2> &unpadded,
                    const zisa::array<azeban::complex_t, 2> &padded) {
  std::map<std::array<long, 2>, azeban::complex_t> m;
  for (zisa::int_t i = 0; i < unpadded.shape(0); ++i) {
    for (zisa::int_t j = 0; j < unpadded.shape(1); ++j) {
      const std::array<long, 2> kvec{to_real_k(i, unpadded.shape(0)),
                                     to_real_k(j, unpadded.shape(0))};
      m[kvec] = unpadded(i, j);
    }
  }
  for (zisa::int_t i = 0; i < padded.shape(0); ++i) {
    for (zisa::int_t j = 0; j < padded.shape(1); ++j) {
      std::array<long, 2> kvec;
      kvec[0] = to_real_k(i, padded.shape(0));
      kvec[1] = to_real_k(j, padded.shape(0));
      if (const auto it = m.find(kvec); it != m.end()) {
        REQUIRE(it->second == padded(i, j));
      } else {
        REQUIRE(padded(i, j) == 0);
      }
    }
  }
}

static void
test_is_zero_padded(const zisa::array<azeban::complex_t, 3> &unpadded,
                    const zisa::array<azeban::complex_t, 3> &padded) {
  std::map<std::array<long, 3>, azeban::complex_t> m;
  for (zisa::int_t i = 0; i < unpadded.shape(0); ++i) {
    for (zisa::int_t j = 0; j < unpadded.shape(1); ++j) {
      for (zisa::int_t k = 0; k < unpadded.shape(2); ++k) {
        const std::array<long, 3> kvec{to_real_k(i, unpadded.shape(0)),
                                       to_real_k(j, unpadded.shape(0)),
                                       to_real_k(k, unpadded.shape(0))};
        m[kvec] = unpadded(i, j, k);
      }
    }
  }
  for (zisa::int_t i = 0; i < padded.shape(0); ++i) {
    for (zisa::int_t j = 0; j < padded.shape(1); ++j) {
      for (zisa::int_t k = 0; k < padded.shape(2); ++k) {
        const std::array<long, 3> kvec{to_real_k(i, padded.shape(0)),
                                       to_real_k(j, padded.shape(0)),
                                       to_real_k(k, padded.shape(0))};
        if (const auto it = m.find(kvec); it != m.end()) {
          REQUIRE(it->second == padded(i, j, k));
        } else {
          REQUIRE(padded(i, j, k) == 0);
        }
      }
    }
  }
}

template <int Dim>
static void test_zero_padding(zisa::int_t N_unpadded,
                              zisa::int_t N_padded,
                              zisa::device_type device) {
  azeban::Grid<Dim> grid(N_unpadded, N_padded);
  zisa::shape_t<Dim + 1> shape_unpadded = grid.shape_fourier(1);
  zisa::shape_t<Dim + 1> shape_padded = grid.shape_fourier_pad(1);
  zisa::shape_t<Dim> unpadded;
  zisa::shape_t<Dim> padded;
  for (zisa::int_t i = 0; i < Dim; ++i) {
    unpadded[i] = shape_unpadded[i + 1];
    padded[i] = shape_padded[i + 1];
  }
  zisa::array<azeban::complex_t, Dim> h_unpadded_arr(unpadded);
  zisa::array<azeban::complex_t, Dim> h_padded_arr(padded);
  zisa::array<azeban::complex_t, Dim> unpadded_arr(unpadded, device);
  zisa::array<azeban::complex_t, Dim> padded_arr(padded, device);

  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  for (azeban::complex_t &c : h_unpadded_arr) {
    c.x = dist(rng);
    c.y = dist(rng);
  }

  zisa::copy(unpadded_arr, h_unpadded_arr);
  azeban::copy_to_padded(padded_arr, unpadded_arr, 0);
  zisa::copy(h_padded_arr, padded_arr);

  test_is_zero_padded(h_unpadded_arr, h_padded_arr);
}

TEST_CASE("Zero Padding 1D CPU", "[operations]") {
  for (zisa::int_t N = 8; N <= 1024; N <<= 1) {
    std::cout << "Zero Padding 1D: N = " << N << std::endl;
    test_zero_padding<1>(N, N, zisa::device_type::cpu);
    test_zero_padding<1>(N, 3 * N / 2, zisa::device_type::cpu);
    test_zero_padding<1>(N, 2 * N, zisa::device_type::cpu);
  }
}

TEST_CASE("Zero Padding 1D GPU", "[operations]") {
  for (zisa::int_t N = 8; N <= 1024; N <<= 1) {
    std::cout << "Zero Padding 1D: N = " << N << std::endl;
    test_zero_padding<1>(N, N, zisa::device_type::cuda);
    test_zero_padding<1>(N, 3 * N / 2, zisa::device_type::cuda);
    test_zero_padding<1>(N, 2 * N, zisa::device_type::cuda);
  }
}

TEST_CASE("Zero Padding 2D CPU", "[operations]") {
  for (zisa::int_t N = 8; N <= 1024; N <<= 1) {
    std::cout << "Zero Padding 2D: N = " << N << std::endl;
    test_zero_padding<2>(N, N, zisa::device_type::cpu);
    test_zero_padding<2>(N, 3 * N / 2, zisa::device_type::cpu);
    test_zero_padding<2>(N, 2 * N, zisa::device_type::cpu);
  }
}

TEST_CASE("Zero Padding 2D GPU", "[operations]") {
  for (zisa::int_t N = 8; N <= 1024; N <<= 1) {
    std::cout << "Zero Padding 2D: N = " << N << std::endl;
    test_zero_padding<2>(N, N, zisa::device_type::cuda);
    test_zero_padding<2>(N, 3 * N / 2, zisa::device_type::cuda);
    test_zero_padding<2>(N, 2 * N, zisa::device_type::cuda);
  }
}

TEST_CASE("Zero Padding 3D CPU", "[operations]") {
  for (zisa::int_t N = 8; N <= 64; N <<= 1) {
    std::cout << "Zero Padding 3D: N = " << N << std::endl;
    test_zero_padding<3>(N, N, zisa::device_type::cpu);
    test_zero_padding<3>(N, 3 * N / 2, zisa::device_type::cpu);
    test_zero_padding<3>(N, 2 * N, zisa::device_type::cpu);
  }
}

TEST_CASE("Zero Padding 3D GPU", "[operations]") {
  for (zisa::int_t N = 8; N <= 64; N <<= 1) {
    std::cout << "Zero Padding 3D: N = " << N << std::endl;
    test_zero_padding<3>(N, N, zisa::device_type::cuda);
    test_zero_padding<3>(N, 3 * N / 2, zisa::device_type::cuda);
    test_zero_padding<3>(N, 2 * N, zisa::device_type::cuda);
  }
}
