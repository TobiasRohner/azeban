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

#include <azeban/grid.hpp>
#include <azeban/operations/fft.hpp>
#include <fmt/core.h>
#include <random>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array_view.hpp>

TEST_CASE("Learn the cufft API", "[cufft]") {

  zisa::int_t n = 128;
  auto shape = zisa::shape_t<1>{n};
  auto u = zisa::array<cufftComplex, 1>(shape);
  auto uhat = zisa::array<cufftComplex, 1>(shape);

  for (zisa::int_t i = 0; i < n; ++i) {
    u[i].x = zisa::cos(2.0 * zisa::pi * i / n);
    u[i].y = 0.0;
  }

  cufftHandle plan;
  auto status = cufftPlan1d(&plan, shape[0], CUFFT_C2C, 1);
  REQUIRE(status == CUFFT_SUCCESS);

  auto d_u = zisa::cuda_array<cufftComplex, 1>(shape);
  auto d_uhat = zisa::cuda_array<cufftComplex, 1>(shape);

  zisa::copy(d_u, u);
  cufftExecC2C(plan, d_u.raw(), d_uhat.raw(), CUFFT_FORWARD);
  zisa::copy(uhat, d_uhat);

  auto d_u2 = zisa::cuda_array<cufftComplex, 1>(shape);
  cufftExecC2C(plan, d_uhat.raw(), d_u2.raw(), CUFFT_INVERSE);

  auto u2 = zisa::array<cufftComplex, 1>(shape);
  zisa::copy(u2, d_u2);
  std::cout << uhat[0].x << ", " << uhat[1].x << ", " << uhat[2].x << "\n";
  std::cout << u2[0].x / n << ", " << u2[1].x / n << ", " << u2[2].x / n
            << "\n";
  std::cout << u[0].x << ", " << u[1].x << ", " << u[2].x << "\n";

  std::cout << (u2[0].x / n) - u[0].x << ", " << (u2[1].x / n) - u[1].x << ", "
            << (u2[2].x / n) - u[2].x << ", " << (u2[3].x / n) - u[3].x << "\n";
}

TEST_CASE("FFTW 1D scalar valued data", "[fft]") {
  zisa::int_t n = 128;
  zisa::shape_t<2> rshape{1, n};
  zisa::shape_t<2> cshape{1, n / 2 + 1};
  zisa::array<azeban::real_t, 2> u(rshape);
  zisa::array<azeban::real_t, 2> u_ref(rshape);
  zisa::array<azeban::complex_t, 2> u_hat(cshape);
  zisa::array<azeban::complex_t, 2> u_hat_ref(cshape);

  std::shared_ptr<azeban::FFT<1>> fft
      = std::make_shared<azeban::FFTWFFT<1>>(u_hat, u);

  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  u_hat_ref[0].x = dist(rng);
  u_hat_ref[0].y = 0;
  for (zisa::int_t k = 1; k < n / 2; ++k) {
    u_hat_ref[k].x = dist(rng);
    u_hat_ref[k].y = dist(rng);
  }
  u_hat_ref[n / 2].x = dist(rng);
  u_hat_ref[n / 2].y = n % 2 ? dist(rng) : 0;
  for (zisa::int_t i = 0; i < n; ++i) {
    u_ref[i] = u_hat_ref[0].x / n;
    for (zisa::int_t k = 1; k < n / 2 + 1; ++k) {
      const azeban::real_t xi
          = 2 * zisa::pi * static_cast<azeban::real_t>(k * i) / n;
      const azeban::complex_t a
          = u_hat_ref[k] * azeban::complex_t(zisa::cos(xi), zisa::sin(xi));
      if (k == n / 2 && n % 2 == 0) {
        u_ref[i] += a.x / n;
      } else {
        u_ref[i] += 2 * a.x / n;
      }
    }
  }
  zisa::copy(u, u_ref);

  fft->forward();

  for (zisa::int_t i = 0; i < n / 2 + 1; ++i) {
    REQUIRE(std::fabs(u_hat[i].x - u_hat_ref[i].x) <= 1e-10);
    REQUIRE(std::fabs(u_hat[i].y - u_hat_ref[i].y) <= 1e-10);
  }

  fft->backward();

  for (zisa::int_t i = 0; i < n; ++i) {
    REQUIRE(std::fabs(u[i] / n - u_ref[i]) <= 1e-10);
  }
}

TEST_CASE("FFTW 1D vector valued data", "[fft]") {
  zisa::int_t n = 128;
  zisa::shape_t<2> rshape{2, n};
  zisa::shape_t<2> cshape{2, n / 2 + 1};
  auto u = zisa::array<azeban::real_t, 2>(rshape);
  auto u_ref = zisa::array<azeban::real_t, 2>(rshape);
  auto u_hat = zisa::array<azeban::complex_t, 2>(cshape);
  auto u_hat_ref = zisa::array<azeban::complex_t, 2>(cshape);

  std::shared_ptr<azeban::FFT<1>> fft
      = std::make_shared<azeban::FFTWFFT<1>>(u_hat, u);

  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  for (zisa::int_t d = 0; d < 2; ++d) {
    u_hat_ref(d, 0).x = dist(rng);
    u_hat_ref(d, 0).y = 0;
    for (zisa::int_t k = 1; k < n / 2; ++k) {
      u_hat_ref(d, k).x = dist(rng);
      u_hat_ref(d, k).y = dist(rng);
    }
    u_hat_ref(d, n / 2).x = dist(rng);
    u_hat_ref(d, n / 2).y = n % 2 ? dist(rng) : 0;
    for (zisa::int_t i = 0; i < n; ++i) {
      u_ref(d, i) = u_hat_ref(d, 0).x / n;
      for (zisa::int_t k = 1; k < n / 2 + 1; ++k) {
        const azeban::real_t xi
            = 2 * zisa::pi * static_cast<azeban::real_t>(k * i) / n;
        const azeban::complex_t a
            = u_hat_ref(d, k) * azeban::complex_t(zisa::cos(xi), zisa::sin(xi));
        if (k == n / 2 && n % 2 == 0) {
          u_ref(d, i) += a.x / n;
        } else {
          u_ref(d, i) += 2 * a.x / n;
        }
      }
    }
  }
  zisa::copy(u, u_ref);

  fft->forward();

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    REQUIRE(std::fabs(u_hat(0, i).x - u_hat_ref(0, i).x) <= 1e-10);
    REQUIRE(std::fabs(u_hat(0, i).y - u_hat_ref(0, i).y) <= 1e-10);
    REQUIRE(std::fabs(u_hat(1, i).x - u_hat_ref(1, i).x) <= 1e-10);
    REQUIRE(std::fabs(u_hat(1, i).y - u_hat_ref(1, i).y) <= 1e-10);
  }

  fft->backward();

  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    REQUIRE(std::fabs(u(0, i) / n - u_ref(0, i)) <= 1e-10);
    REQUIRE(std::fabs(u(1, i) / n - u_ref(1, i)) <= 1e-10);
  }
}

TEST_CASE("FFTW 2D scalar valued data", "[fft]") {
  zisa::int_t n = 128;
  zisa::shape_t<3> rshape{1, n, n};
  zisa::shape_t<3> cshape{1, n, n / 2 + 1};
  auto u = zisa::array<azeban::real_t, 3>(rshape);
  auto u_ref = zisa::array<azeban::real_t, 3>(rshape);
  auto u_hat = zisa::array<azeban::complex_t, 3>(cshape);
  auto u_hat_ref = zisa::array<azeban::complex_t, 3>(cshape);

  std::shared_ptr<azeban::FFT<2>> fft
      = std::make_shared<azeban::FFTWFFT<2>>(u_hat, u);

  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  for (zisa::int_t i = 0; i < n; ++i) {
    if (i < n / 2 + 1) {
      u_hat_ref(0, i, 0).x = dist(rng);
      u_hat_ref(0, i, 0).y
          = (i == 0 || (i == n / 2 && n % 2 == 0)) ? 0 : dist(rng);
    } else {
      u_hat_ref(0, i, 0) = azeban::complex_t(u_hat_ref(0, n - i, 0).x,
                                             -u_hat_ref(0, n - i, 0).y);
    }
    for (zisa::int_t j = 1; j < n / 2; ++j) {
      u_hat_ref(0, i, j).x = dist(rng);
      u_hat_ref(0, i, j).y = dist(rng);
    }
    if (i < n / 2 + 1) {
      u_hat_ref(0, i, n / 2).x = dist(rng);
      u_hat_ref(0, i, n / 2).y
          = (n % 2 == 0 && (i == 0 || (i == n / 2 && n % 2 == 0))) ? 0
                                                                   : dist(rng);
    } else {
      u_hat_ref(0, i, n / 2) = azeban::complex_t(u_hat_ref(0, n - i, n / 2).x,
                                                 -u_hat_ref(0, n - i, n / 2).y);
    }
  }
  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      u_ref(0, i, j) = 0;
      for (zisa::int_t k = 0; k < n; ++k) {
        {
          const azeban::real_t xi
              = 2 * zisa::pi * static_cast<azeban::real_t>(i * k) / n;
          const azeban::real_t eta = 0;
          const azeban::complex_t a
              = u_hat_ref(0, k, 0)
                * azeban::complex_t(zisa::cos(xi + eta), zisa::sin(xi + eta));
          u_ref(0, i, j) += a.x / (n * n);
        }
        for (zisa::int_t l = 1; l < n / 2; ++l) {
          const azeban::real_t xi
              = 2 * zisa::pi * static_cast<azeban::real_t>(i * k) / n;
          const azeban::real_t eta
              = 2 * zisa::pi * static_cast<azeban::real_t>(j * l) / n;
          const azeban::complex_t a
              = u_hat_ref(0, k, l)
                * azeban::complex_t(zisa::cos(xi + eta), zisa::sin(xi + eta));
          u_ref(0, i, j) += 2 * a.x / (n * n);
        }
        {
          const azeban::real_t xi
              = 2 * zisa::pi * static_cast<azeban::real_t>(i * k) / n;
          const azeban::real_t eta
              = 2 * zisa::pi * static_cast<azeban::real_t>(j * (n / 2)) / n;
          const azeban::complex_t a
              = u_hat_ref(0, k, n / 2)
                * azeban::complex_t(zisa::cos(xi + eta), zisa::sin(xi + eta));
          if (n % 2 == 0) {
            u_ref(0, i, j) += a.x / (n * n);
          } else {
            u_ref(0, i, j) += 2 * a.x / (n * n);
          }
        }
      }
    }
  }
  zisa::copy(u, u_ref);

  fft->forward();

  for (zisa::int_t k = 0; k < cshape[1]; ++k) {
    for (zisa::int_t l = 0; l < cshape[2]; ++l) {
      REQUIRE(std::fabs(u_hat(0, k, l).x - u_hat_ref(0, k, l).x) <= 1e-10);
      REQUIRE(std::fabs(u_hat(0, k, l).y - u_hat_ref(0, k, l).y) <= 1e-10);
    }
  }

  fft->backward();

  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      REQUIRE(std::fabs(u(0, i, j) / (n * n) - u_ref(0, i, j)) <= 1e-8);
    }
  }
}

TEST_CASE("FFTW 2D vector valued data", "[fft]") {
  zisa::int_t n = 128;
  zisa::shape_t<3> rshape{2, n, n};
  zisa::shape_t<3> cshape{2, n, n / 2 + 1};
  auto u = zisa::array<azeban::real_t, 3>(rshape);
  auto u_ref = zisa::array<azeban::real_t, 3>(rshape);
  auto u_hat = zisa::array<azeban::complex_t, 3>(cshape);
  auto u_hat_ref = zisa::array<azeban::complex_t, 3>(cshape);

  std::shared_ptr<azeban::FFT<2>> fft
      = std::make_shared<azeban::FFTWFFT<2>>(u_hat, u);

  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  for (zisa::int_t d = 0; d < 2; ++d) {
    for (zisa::int_t i = 0; i < n; ++i) {
      if (i < n / 2 + 1) {
        u_hat_ref(d, i, 0).x = dist(rng);
        u_hat_ref(d, i, 0).y
            = (i == 0 || (i == n / 2 && n % 2 == 0)) ? 0 : dist(rng);
      } else {
        u_hat_ref(d, i, 0) = azeban::complex_t(u_hat_ref(d, n - i, 0).x,
                                               -u_hat_ref(d, n - i, 0).y);
      }
      for (zisa::int_t j = 1; j < n / 2; ++j) {
        u_hat_ref(d, i, j).x = dist(rng);
        u_hat_ref(d, i, j).y = dist(rng);
      }
      if (i < n / 2 + 1) {
        u_hat_ref(d, i, n / 2).x = dist(rng);
        u_hat_ref(d, i, n / 2).y
            = (n % 2 == 0 && (i == 0 || (i == n / 2 && n % 2 == 0)))
                  ? 0
                  : dist(rng);
      } else {
        u_hat_ref(d, i, n / 2) = azeban::complex_t(
            u_hat_ref(d, n - i, n / 2).x, -u_hat_ref(d, n - i, n / 2).y);
      }
    }
    for (zisa::int_t i = 0; i < n; ++i) {
      for (zisa::int_t j = 0; j < n; ++j) {
        u_ref(d, i, j) = 0;
        for (zisa::int_t k = 0; k < n; ++k) {
          {
            const azeban::real_t xi
                = 2 * zisa::pi * static_cast<azeban::real_t>(i * k) / n;
            const azeban::real_t eta = 0;
            const azeban::complex_t a
                = u_hat_ref(d, k, 0)
                  * azeban::complex_t(zisa::cos(xi + eta), zisa::sin(xi + eta));
            u_ref(d, i, j) += a.x / (n * n);
          }
          for (zisa::int_t l = 1; l < n / 2; ++l) {
            const azeban::real_t xi
                = 2 * zisa::pi * static_cast<azeban::real_t>(i * k) / n;
            const azeban::real_t eta
                = 2 * zisa::pi * static_cast<azeban::real_t>(j * l) / n;
            const azeban::complex_t a
                = u_hat_ref(d, k, l)
                  * azeban::complex_t(zisa::cos(xi + eta), zisa::sin(xi + eta));
            u_ref(d, i, j) += 2 * a.x / (n * n);
          }
          {
            const azeban::real_t xi
                = 2 * zisa::pi * static_cast<azeban::real_t>(i * k) / n;
            const azeban::real_t eta
                = 2 * zisa::pi * static_cast<azeban::real_t>(j * (n / 2)) / n;
            const azeban::complex_t a
                = u_hat_ref(d, k, n / 2)
                  * azeban::complex_t(zisa::cos(xi + eta), zisa::sin(xi + eta));
            if (n % 2 == 0) {
              u_ref(d, i, j) += a.x / (n * n);
            } else {
              u_ref(d, i, j) += 2 * a.x / (n * n);
            }
          }
        }
      }
    }
  }
  zisa::copy(u, u_ref);

  fft->forward();

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    for (zisa::int_t j = 0; j < cshape[2]; ++j) {
      REQUIRE(std::fabs(u_hat(0, i, j).x - u_hat_ref(0, i, j).x) <= 1e-10);
      REQUIRE(std::fabs(u_hat(0, i, j).x - u_hat_ref(0, i, j).x) <= 1e-10);
      REQUIRE(std::fabs(u_hat(1, i, j).y - u_hat_ref(1, i, j).y) <= 1e-10);
      REQUIRE(std::fabs(u_hat(1, i, j).y - u_hat_ref(1, i, j).y) <= 1e-10);
    }
  }

  fft->backward();

  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      REQUIRE(std::fabs(u(0, i, j) / (n * n) - u_ref(0, i, j)) <= 1e-8);
      REQUIRE(std::fabs(u(1, i, j) / (n * n) - u_ref(1, i, j)) <= 1e-8);
    }
  }
}

TEST_CASE("FFTW 3D scalar valued data", "[fft]") {
  zisa::int_t n = 128;
  zisa::shape_t<4> rshape{1, n, n, n};
  zisa::shape_t<4> cshape{1, n, n, n / 2 + 1};
  auto u = zisa::array<azeban::real_t, 4>(rshape);
  auto u_hat = zisa::array<azeban::complex_t, 4>(cshape);

  std::shared_ptr<azeban::FFT<3>> fft
      = std::make_shared<azeban::FFTWFFT<3>>(u_hat, u);

  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      for (zisa::int_t k = 0; k < n; ++k) {
        u(0, i, j, k) = zisa::cos(2.0 * zisa::pi * (i + j + k) / n);
      }
    }
  }

  fft->forward();

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    for (zisa::int_t j = 0; j < cshape[2]; ++j) {
      for (zisa::int_t k = 0; k < cshape[3]; ++k) {
        const int i_ = i >= n / 2 + 1 ? zisa::integer_cast<int>(i) - n
                                      : zisa::integer_cast<int>(i);
        const int j_ = j >= n / 2 + 1 ? zisa::integer_cast<int>(j) - n
                                      : zisa::integer_cast<int>(j);
        const int k_ = k >= n / 2 + 1 ? zisa::integer_cast<int>(k) - n
                                      : zisa::integer_cast<int>(k);
        azeban::complex_t expected;
        expected.x = (i_ == 1 && j_ == 1 && k_ == 1)
                             || (i_ == -1 && j_ == -1 && k_ == -1)
                         ? n * n * n / 2.0
                         : 0.0;
        expected.y = 0;
        REQUIRE(std::fabs(u_hat(0, i, j, k).x - expected.x) <= 1e-9);
        REQUIRE(std::fabs(u_hat(0, i, j, k).y - expected.y) <= 1e-9);
      }
    }
  }

  fft->backward();

  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      for (zisa::int_t k = 0; k < n; ++k) {
        const azeban::real_t expected
            = n * n * n * zisa::cos(2.0 * zisa::pi * (i + j + k) / n);
        REQUIRE(std::fabs(u(0, i, j, k) - expected) <= 1e-8);
      }
    }
  }
}

TEST_CASE("FFTW 3D vector valued data", "[fft]") {
  zisa::int_t n = 128;
  zisa::shape_t<4> rshape{3, n, n, n};
  zisa::shape_t<4> cshape{3, n, n, n / 2 + 1};
  auto u = zisa::array<azeban::real_t, 4>(rshape);
  auto u_hat = zisa::array<azeban::complex_t, 4>(cshape);

  std::shared_ptr<azeban::FFT<3>> fft
      = std::make_shared<azeban::FFTWFFT<3>>(u_hat, u);

  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      for (zisa::int_t k = 0; k < n; ++k) {
        u(0, i, j, k) = zisa::cos(2.0 * zisa::pi * (i + j + k) / n);
        u(1, i, j, k) = zisa::cos(4.0 * zisa::pi * (i + j + k) / n);
        u(2, i, j, k) = zisa::cos(6.0 * zisa::pi * (i + j + k) / n);
      }
    }
  }

  fft->forward();

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    for (zisa::int_t j = 0; j < cshape[2]; ++j) {
      for (zisa::int_t k = 0; k < cshape[3]; ++k) {
        const int i_ = i >= n / 2 + 1 ? zisa::integer_cast<int>(i) - n
                                      : zisa::integer_cast<int>(i);
        const int j_ = j >= n / 2 + 1 ? zisa::integer_cast<int>(j) - n
                                      : zisa::integer_cast<int>(j);
        const int k_ = k >= n / 2 + 1 ? zisa::integer_cast<int>(k) - n
                                      : zisa::integer_cast<int>(k);
        azeban::complex_t expected_0;
        azeban::complex_t expected_1;
        azeban::complex_t expected_2;
        expected_0.x = (i_ == 1 && j_ == 1 && k_ == 1)
                               || (i_ == -1 && j_ == -1 && k_ == -1)
                           ? n * n * n / 2.0
                           : 0.0;
        expected_0.y = 0;
        expected_1.x = (i_ == 2 && j_ == 2 && k_ == 2)
                               || (i_ == -2 && j_ == -2 && k_ == -2)
                           ? n * n * n / 2.0
                           : 0.0;
        expected_1.y = 0;
        expected_2.x = (i_ == 3 && j_ == 3 && k_ == 3)
                               || (i_ == -3 && j_ == -3 && k_ == -3)
                           ? n * n * n / 2.0
                           : 0.0;
        expected_2.y = 0;
        REQUIRE(std::fabs(u_hat(0, i, j, k).x - expected_0.x) <= 1e-8);
        REQUIRE(std::fabs(u_hat(0, i, j, k).y - expected_0.y) <= 1e-8);
        REQUIRE(std::fabs(u_hat(1, i, j, k).x - expected_1.x) <= 1e-8);
        REQUIRE(std::fabs(u_hat(1, i, j, k).y - expected_1.y) <= 1e-8);
        REQUIRE(std::fabs(u_hat(2, i, j, k).x - expected_2.x) <= 1e-8);
        REQUIRE(std::fabs(u_hat(2, i, j, k).y - expected_2.y) <= 1e-8);
      }
    }
  }

  fft->backward();

  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      for (zisa::int_t k = 0; k < n; ++k) {
        const azeban::real_t expected1
            = n * n * n * zisa::cos(2.0 * zisa::pi * (i + j + k) / n);
        const azeban::real_t expected2
            = n * n * n * zisa::cos(4.0 * zisa::pi * (i + j + k) / n);
        const azeban::real_t expected3
            = n * n * n * zisa::cos(6.0 * zisa::pi * (i + j + k) / n);
        REQUIRE(std::fabs(u(0, i, j, k) - expected1) <= 1e-8);
        REQUIRE(std::fabs(u(1, i, j, k) - expected2) <= 1e-8);
        REQUIRE(std::fabs(u(2, i, j, k) - expected3) <= 1e-8);
      }
    }
  }
}

TEST_CASE("cuFFT 1D scalar valued data", "[cufft]") {
  zisa::int_t n = 128;
  zisa::shape_t<2> rshape{1, n};
  zisa::shape_t<2> cshape{1, n / 2 + 1};
  auto h_u = zisa::array<azeban::real_t, 2>(rshape);
  auto h_u_ref = zisa::array<azeban::real_t, 2>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 2>(cshape);
  auto h_u_hat_ref = zisa::array<azeban::complex_t, 2>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 2>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 2>(cshape);

  std::shared_ptr<azeban::FFT<1>> fft
      = std::make_shared<azeban::CUFFT<1>>(d_u_hat, d_u);

  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  h_u_hat_ref[0].x = dist(rng);
  h_u_hat_ref[0].y = 0;
  for (zisa::int_t k = 1; k < n / 2; ++k) {
    h_u_hat_ref[k].x = dist(rng);
    h_u_hat_ref[k].y = dist(rng);
  }
  h_u_hat_ref[n / 2].x = dist(rng);
  h_u_hat_ref[n / 2].y = n % 2 ? dist(rng) : 0;
  for (zisa::int_t i = 0; i < n; ++i) {
    h_u_ref[i] = h_u_hat_ref[0].x / n;
    for (zisa::int_t k = 1; k < n / 2 + 1; ++k) {
      const azeban::real_t xi
          = 2 * zisa::pi * static_cast<azeban::real_t>(k * i) / n;
      const azeban::complex_t a
          = h_u_hat_ref[k] * azeban::complex_t(zisa::cos(xi), zisa::sin(xi));
      if (k == n / 2 && n % 2 == 0) {
        h_u_ref[i] += a.x / n;
      } else {
        h_u_ref[i] += 2 * a.x / n;
      }
    }
  }
  zisa::copy(h_u, h_u_ref);

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    REQUIRE(std::fabs(h_u_hat[i].x - h_u_hat_ref[i].x) <= 1e-10);
    REQUIRE(std::fabs(h_u_hat[i].y - h_u_hat_ref[i].y) <= 1e-10);
  }
  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    REQUIRE(std::fabs(h_u[i] / n - h_u_ref[i]) <= 1e-10);
  }
}

TEST_CASE("cuFFT 1D vector valued data", "[cufft]") {
  zisa::int_t n = 128;
  zisa::shape_t<2> rshape{2, n};
  zisa::shape_t<2> cshape{2, n / 2 + 1};
  auto h_u = zisa::array<azeban::real_t, 2>(rshape);
  auto h_u_ref = zisa::array<azeban::real_t, 2>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 2>(cshape);
  auto h_u_hat_ref = zisa::array<azeban::complex_t, 2>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 2>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 2>(cshape);

  std::shared_ptr<azeban::FFT<1>> fft
      = std::make_shared<azeban::CUFFT<1>>(d_u_hat, d_u);

  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  for (zisa::int_t d = 0; d < 2; ++d) {
    h_u_hat_ref(d, 0).x = dist(rng);
    h_u_hat_ref(d, 0).y = 0;
    for (zisa::int_t k = 1; k < n / 2; ++k) {
      h_u_hat_ref(d, k).x = dist(rng);
      h_u_hat_ref(d, k).y = dist(rng);
    }
    h_u_hat_ref(d, n / 2).x = dist(rng);
    h_u_hat_ref(d, n / 2).y = n % 2 ? dist(rng) : 0;
    for (zisa::int_t i = 0; i < n; ++i) {
      h_u_ref(d, i) = h_u_hat_ref(d, 0).x / n;
      for (zisa::int_t k = 1; k < n / 2 + 1; ++k) {
        const azeban::real_t xi
            = 2 * zisa::pi * static_cast<azeban::real_t>(k * i) / n;
        const azeban::complex_t a
            = h_u_hat_ref(d, k)
              * azeban::complex_t(zisa::cos(xi), zisa::sin(xi));
        if (k == n / 2 && n % 2 == 0) {
          h_u_ref(d, i) += a.x / n;
        } else {
          h_u_ref(d, i) += 2 * a.x / n;
        }
      }
    }
  }
  zisa::copy(h_u, h_u_ref);

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    REQUIRE(std::fabs(h_u_hat(0, i).x - h_u_hat_ref(0, i).x) <= 1e-10);
    REQUIRE(std::fabs(h_u_hat(0, i).y - h_u_hat_ref(0, i).y) <= 1e-10);
    REQUIRE(std::fabs(h_u_hat(1, i).x - h_u_hat_ref(1, i).x) <= 1e-10);
    REQUIRE(std::fabs(h_u_hat(1, i).y - h_u_hat_ref(1, i).y) <= 1e-10);
  }
  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    REQUIRE(std::fabs(h_u(0, i) / n - h_u_ref(0, i)) <= 1e-10);
    REQUIRE(std::fabs(h_u(1, i) / n - h_u_ref(1, i)) <= 1e-10);
  }
}

TEST_CASE("cuFFT 2D scalar valued data", "[cufft]") {
  zisa::int_t n = 128;
  zisa::shape_t<3> rshape{1, n, n};
  zisa::shape_t<3> cshape{1, n, n / 2 + 1};
  auto h_u = zisa::array<azeban::real_t, 3>(rshape);
  auto h_u_ref = zisa::array<azeban::real_t, 3>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 3>(cshape);
  auto h_u_hat_ref = zisa::array<azeban::complex_t, 3>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 3>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 3>(cshape);

  std::shared_ptr<azeban::FFT<2>> fft
      = std::make_shared<azeban::CUFFT<2>>(d_u_hat, d_u);

  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  for (zisa::int_t i = 0; i < n; ++i) {
    if (i < n / 2 + 1) {
      h_u_hat_ref(0, i, 0).x = dist(rng);
      h_u_hat_ref(0, i, 0).y
          = (i == 0 || (i == n / 2 && n % 2 == 0)) ? 0 : dist(rng);
    } else {
      h_u_hat_ref(0, i, 0) = azeban::complex_t(h_u_hat_ref(0, n - i, 0).x,
                                               -h_u_hat_ref(0, n - i, 0).y);
    }
    for (zisa::int_t j = 1; j < n / 2; ++j) {
      h_u_hat_ref(0, i, j).x = dist(rng);
      h_u_hat_ref(0, i, j).y = dist(rng);
    }
    if (i < n / 2 + 1) {
      h_u_hat_ref(0, i, n / 2).x = dist(rng);
      h_u_hat_ref(0, i, n / 2).y
          = (n % 2 == 0 && (i == 0 || (i == n / 2 && n % 2 == 0))) ? 0
                                                                   : dist(rng);
    } else {
      h_u_hat_ref(0, i, n / 2) = azeban::complex_t(
          h_u_hat_ref(0, n - i, n / 2).x, -h_u_hat_ref(0, n - i, n / 2).y);
    }
  }
  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      h_u_ref(0, i, j) = 0;
      for (zisa::int_t k = 0; k < n; ++k) {
        {
          const azeban::real_t xi
              = 2 * zisa::pi * static_cast<azeban::real_t>(i * k) / n;
          const azeban::real_t eta = 0;
          const azeban::complex_t a
              = h_u_hat_ref(0, k, 0)
                * azeban::complex_t(zisa::cos(xi + eta), zisa::sin(xi + eta));
          h_u_ref(0, i, j) += a.x / (n * n);
        }
        for (zisa::int_t l = 1; l < n / 2; ++l) {
          const azeban::real_t xi
              = 2 * zisa::pi * static_cast<azeban::real_t>(i * k) / n;
          const azeban::real_t eta
              = 2 * zisa::pi * static_cast<azeban::real_t>(j * l) / n;
          const azeban::complex_t a
              = h_u_hat_ref(0, k, l)
                * azeban::complex_t(zisa::cos(xi + eta), zisa::sin(xi + eta));
          h_u_ref(0, i, j) += 2 * a.x / (n * n);
        }
        {
          const azeban::real_t xi
              = 2 * zisa::pi * static_cast<azeban::real_t>(i * k) / n;
          const azeban::real_t eta
              = 2 * zisa::pi * static_cast<azeban::real_t>(j * (n / 2)) / n;
          const azeban::complex_t a
              = h_u_hat_ref(0, k, n / 2)
                * azeban::complex_t(zisa::cos(xi + eta), zisa::sin(xi + eta));
          if (n % 2 == 0) {
            h_u_ref(0, i, j) += a.x / (n * n);
          } else {
            h_u_ref(0, i, j) += 2 * a.x / (n * n);
          }
        }
      }
    }
  }
  zisa::copy(h_u, h_u_ref);

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t k = 0; k < cshape[1]; ++k) {
    for (zisa::int_t l = 0; l < cshape[2]; ++l) {
      REQUIRE(std::fabs(h_u_hat(0, k, l).x - h_u_hat_ref(0, k, l).x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(0, k, l).y - h_u_hat_ref(0, k, l).y) <= 1e-10);
    }
  }
  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      REQUIRE(std::fabs(h_u(0, i, j) / (n * n) - h_u_ref(0, i, j)) <= 1e-8);
    }
  }
}

TEST_CASE("cuFFT 2D vector valued data", "[cufft]") {
  zisa::int_t n = 128;
  zisa::shape_t<3> rshape{2, n, n};
  zisa::shape_t<3> cshape{2, n, n / 2 + 1};
  auto h_u = zisa::array<azeban::real_t, 3>(rshape);
  auto h_u_ref = zisa::array<azeban::real_t, 3>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 3>(cshape);
  auto h_u_hat_ref = zisa::array<azeban::complex_t, 3>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 3>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 3>(cshape);

  std::shared_ptr<azeban::FFT<2>> fft
      = std::make_shared<azeban::CUFFT<2>>(d_u_hat, d_u);

  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  for (zisa::int_t d = 0; d < 2; ++d) {
    for (zisa::int_t i = 0; i < n; ++i) {
      if (i < n / 2 + 1) {
        h_u_hat_ref(d, i, 0).x = dist(rng);
        h_u_hat_ref(d, i, 0).y
            = (i == 0 || (i == n / 2 && n % 2 == 0)) ? 0 : dist(rng);
      } else {
        h_u_hat_ref(d, i, 0) = azeban::complex_t(h_u_hat_ref(d, n - i, 0).x,
                                                 -h_u_hat_ref(d, n - i, 0).y);
      }
      for (zisa::int_t j = 1; j < n / 2; ++j) {
        h_u_hat_ref(d, i, j).x = dist(rng);
        h_u_hat_ref(d, i, j).y = dist(rng);
      }
      if (i < n / 2 + 1) {
        h_u_hat_ref(d, i, n / 2).x = dist(rng);
        h_u_hat_ref(d, i, n / 2).y
            = (n % 2 == 0 && (i == 0 || (i == n / 2 && n % 2 == 0)))
                  ? 0
                  : dist(rng);
      } else {
        h_u_hat_ref(d, i, n / 2) = azeban::complex_t(
            h_u_hat_ref(d, n - i, n / 2).x, -h_u_hat_ref(d, n - i, n / 2).y);
      }
    }
    for (zisa::int_t i = 0; i < n; ++i) {
      for (zisa::int_t j = 0; j < n; ++j) {
        h_u_ref(d, i, j) = 0;
        for (zisa::int_t k = 0; k < n; ++k) {
          {
            const azeban::real_t xi
                = 2 * zisa::pi * static_cast<azeban::real_t>(i * k) / n;
            const azeban::real_t eta = 0;
            const azeban::complex_t a
                = h_u_hat_ref(d, k, 0)
                  * azeban::complex_t(zisa::cos(xi + eta), zisa::sin(xi + eta));
            h_u_ref(d, i, j) += a.x / (n * n);
          }
          for (zisa::int_t l = 1; l < n / 2; ++l) {
            const azeban::real_t xi
                = 2 * zisa::pi * static_cast<azeban::real_t>(i * k) / n;
            const azeban::real_t eta
                = 2 * zisa::pi * static_cast<azeban::real_t>(j * l) / n;
            const azeban::complex_t a
                = h_u_hat_ref(d, k, l)
                  * azeban::complex_t(zisa::cos(xi + eta), zisa::sin(xi + eta));
            h_u_ref(d, i, j) += 2 * a.x / (n * n);
          }
          {
            const azeban::real_t xi
                = 2 * zisa::pi * static_cast<azeban::real_t>(i * k) / n;
            const azeban::real_t eta
                = 2 * zisa::pi * static_cast<azeban::real_t>(j * (n / 2)) / n;
            const azeban::complex_t a
                = h_u_hat_ref(d, k, n / 2)
                  * azeban::complex_t(zisa::cos(xi + eta), zisa::sin(xi + eta));
            if (n % 2 == 0) {
              h_u_ref(d, i, j) += a.x / (n * n);
            } else {
              h_u_ref(d, i, j) += 2 * a.x / (n * n);
            }
          }
        }
      }
    }
  }
  zisa::copy(h_u, h_u_ref);

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t k = 0; k < cshape[1]; ++k) {
    for (zisa::int_t l = 0; l < cshape[2]; ++l) {
      REQUIRE(std::fabs(h_u_hat(0, k, l).x - h_u_hat_ref(0, k, l).x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(0, k, l).y - h_u_hat_ref(0, k, l).y) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(1, k, l).x - h_u_hat_ref(1, k, l).x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(1, k, l).y - h_u_hat_ref(1, k, l).y) <= 1e-10);
    }
  }
  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      REQUIRE(std::fabs(h_u(0, i, j) / (n * n) - h_u_ref(0, i, j)) <= 1e-8);
      REQUIRE(std::fabs(h_u(1, i, j) / (n * n) - h_u_ref(1, i, j)) <= 1e-8);
    }
  }
}

TEST_CASE("cuFFT 3D scalar valued data", "[cufft]") {
  zisa::int_t n = 128;
  zisa::shape_t<4> rshape{1, n, n, n};
  zisa::shape_t<4> cshape{1, n, n, n / 2 + 1};
  auto h_u = zisa::array<azeban::real_t, 4>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 4>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 4>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 4>(cshape);

  std::shared_ptr<azeban::FFT<3>> fft
      = std::make_shared<azeban::CUFFT<3>>(d_u_hat, d_u);

  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      for (zisa::int_t k = 0; k < n; ++k) {
        h_u(0, i, j, k) = zisa::cos(2.0 * zisa::pi * (i + j + k) / n);
      }
    }
  }

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    for (zisa::int_t j = 0; j < cshape[2]; ++j) {
      for (zisa::int_t k = 0; k < cshape[3]; ++k) {
        const int i_ = i >= n / 2 + 1 ? zisa::integer_cast<int>(i) - n
                                      : zisa::integer_cast<int>(i);
        const int j_ = j >= n / 2 + 1 ? zisa::integer_cast<int>(j) - n
                                      : zisa::integer_cast<int>(j);
        const int k_ = k >= n / 2 + 1 ? zisa::integer_cast<int>(k) - n
                                      : zisa::integer_cast<int>(k);
        azeban::complex_t expected;
        expected.x = (i_ == 1 && j_ == 1 && k_ == 1)
                             || (i_ == -1 && j_ == -1 && k_ == -1)
                         ? n * n * n / 2.0
                         : 0.0;
        expected.y = 0;
        REQUIRE(std::fabs(h_u_hat(0, i, j, k).x - expected.x) <= 1e-9);
        REQUIRE(std::fabs(h_u_hat(0, i, j, k).y - expected.y) <= 1e-9);
      }
    }
  }
}

TEST_CASE("cuFFT 3D vector valued data", "[cufft]") {
  zisa::int_t n = 128;
  zisa::shape_t<4> rshape{3, n, n, n};
  zisa::shape_t<4> cshape{3, n, n, n / 2 + 1};
  auto h_u = zisa::array<azeban::real_t, 4>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 4>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 4>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 4>(cshape);

  std::shared_ptr<azeban::FFT<3>> fft
      = std::make_shared<azeban::CUFFT<3>>(d_u_hat, d_u);

  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      for (zisa::int_t k = 0; k < n; ++k) {
        h_u(0, i, j, k) = zisa::cos(2.0 * zisa::pi * (i + j + k) / n);
        h_u(1, i, j, k) = zisa::cos(4.0 * zisa::pi * (i + j + k) / n);
        h_u(2, i, j, k) = zisa::cos(6.0 * zisa::pi * (i + j + k) / n);
      }
    }
  }

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    for (zisa::int_t j = 0; j < cshape[2]; ++j) {
      for (zisa::int_t k = 0; k < cshape[3]; ++k) {
        const int i_ = i >= n / 2 + 1 ? zisa::integer_cast<int>(i) - n
                                      : zisa::integer_cast<int>(i);
        const int j_ = j >= n / 2 + 1 ? zisa::integer_cast<int>(j) - n
                                      : zisa::integer_cast<int>(j);
        const int k_ = k >= n / 2 + 1 ? zisa::integer_cast<int>(k) - n
                                      : zisa::integer_cast<int>(k);
        azeban::complex_t expected_0;
        azeban::complex_t expected_1;
        azeban::complex_t expected_2;
        expected_0.x = (i_ == 1 && j_ == 1 && k_ == 1)
                               || (i_ == -1 && j_ == -1 && k_ == -1)
                           ? n * n * n / 2.0
                           : 0.0;
        expected_0.y = 0;
        expected_1.x = (i_ == 2 && j_ == 2 && k_ == 2)
                               || (i_ == -2 && j_ == -2 && k_ == -2)
                           ? n * n * n / 2.0
                           : 0.0;
        expected_1.y = 0;
        expected_2.x = (i_ == 3 && j_ == 3 && k_ == 3)
                               || (i_ == -3 && j_ == -3 && k_ == -3)
                           ? n * n * n / 2.0
                           : 0.0;
        expected_2.y = 0;
        REQUIRE(std::fabs(h_u_hat(0, i, j, k).x - expected_0.x) <= 1e-8);
        REQUIRE(std::fabs(h_u_hat(0, i, j, k).y - expected_0.y) <= 1e-8);
        REQUIRE(std::fabs(h_u_hat(1, i, j, k).x - expected_1.x) <= 1e-8);
        REQUIRE(std::fabs(h_u_hat(1, i, j, k).y - expected_1.y) <= 1e-8);
        REQUIRE(std::fabs(h_u_hat(2, i, j, k).x - expected_2.x) <= 1e-8);
        REQUIRE(std::fabs(h_u_hat(2, i, j, k).y - expected_2.y) <= 1e-8);
      }
    }
  }
}

#if AZEBAN_HAS_MPI
TEST_CASE("cuFFT MPI 2d scalar valued data", "[mpi]") {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  zisa::int_t n = 128;
  azeban::Grid<2> grid(n);
  zisa::shape_t<3> rshape = grid.shape_phys(1, MPI_COMM_WORLD);
  zisa::shape_t<3> cshape = grid.shape_fourier(1, MPI_COMM_WORLD);
  auto h_u = zisa::array<azeban::real_t, 3>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 3>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 3>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 3>(cshape);

  std::shared_ptr<azeban::FFT<2>> fft
      = std::make_shared<azeban::CUFFT_MPI<2>>(d_u_hat, d_u, MPI_COMM_WORLD);

  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    const zisa::int_t i_ = grid.i_phys(i, MPI_COMM_WORLD);
    for (zisa::int_t j = 0; j < rshape[2]; ++j) {
      h_u(0, i, j) = zisa::cos(2.0 * zisa::pi * (i_ + j) / n);
    }
  }

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    for (zisa::int_t j = 0; j < cshape[2]; ++j) {
      const zisa::int_t i__ = grid.i_fourier(i, MPI_COMM_WORLD);
      const int i_ = i__ >= n / 2 + 1 ? zisa::integer_cast<int>(i__) - n
                                      : zisa::integer_cast<int>(i__);
      const int j_ = j >= n / 2 + 1 ? zisa::integer_cast<int>(j) - n
                                    : zisa::integer_cast<int>(j);
      azeban::complex_t expected;
      expected.x
          = (i_ == 1 && j_ == 1) || (i_ == -1 && j_ == -1) ? n * n / 2.0 : 0.0;
      expected.y = 0;
      REQUIRE(std::fabs(h_u_hat(0, i, j).x - expected.x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(0, i, j).y - expected.y) <= 1e-10);
    }
  }

  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    for (zisa::int_t j = 0; j < rshape[2]; ++j) {
      const zisa::int_t i_ = grid.i_phys(i, MPI_COMM_WORLD);
      const azeban::real_t expected
          = n * n * zisa::cos(2.0 * zisa::pi * (i_ + j) / n);
      REQUIRE(std::fabs(h_u(0, i, j) - expected) <= 1e-8);
    }
  }
}

TEST_CASE("cuFFT MPI 2d vector valued data", "[mpi]") {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  zisa::int_t n = 128;
  azeban::Grid<2> grid(n);
  zisa::shape_t<3> rshape = grid.shape_phys(2, MPI_COMM_WORLD);
  zisa::shape_t<3> cshape = grid.shape_fourier(2, MPI_COMM_WORLD);
  auto h_u = zisa::array<azeban::real_t, 3>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 3>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 3>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 3>(cshape);

  std::shared_ptr<azeban::FFT<2>> fft
      = std::make_shared<azeban::CUFFT_MPI<2>>(d_u_hat, d_u, MPI_COMM_WORLD);

  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    const zisa::int_t i_ = grid.i_phys(i, MPI_COMM_WORLD);
    for (zisa::int_t j = 0; j < rshape[2]; ++j) {
      h_u(0, i, j) = zisa::cos(2.0 * zisa::pi * (i_ + j) / n);
      h_u(1, i, j) = zisa::cos(4.0 * zisa::pi * (i_ + j) / n);
    }
  }

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    for (zisa::int_t j = 0; j < cshape[2]; ++j) {
      const zisa::int_t i__ = grid.i_fourier(i, MPI_COMM_WORLD);
      const int i_ = i__ >= n / 2 + 1 ? zisa::integer_cast<int>(i__) - n
                                      : zisa::integer_cast<int>(i__);
      const int j_ = j >= n / 2 + 1 ? zisa::integer_cast<int>(j) - n
                                    : zisa::integer_cast<int>(j);
      azeban::complex_t expected_0;
      expected_0.x
          = (i_ == 1 && j_ == 1) || (i_ == -1 && j_ == -1) ? n * n / 2.0 : 0.0;
      expected_0.y = 0;
      azeban::complex_t expected_1;
      expected_1.x
          = (i_ == 2 && j_ == 2) || (i_ == -2 && j_ == -2) ? n * n / 2.0 : 0.0;
      expected_1.y = 0;
      REQUIRE(std::fabs(h_u_hat(0, i, j).x - expected_0.x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(0, i, j).x - expected_0.x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(1, i, j).y - expected_1.y) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(1, i, j).y - expected_1.y) <= 1e-10);
    }
  }

  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    for (zisa::int_t j = 0; j < rshape[2]; ++j) {
      const zisa::int_t i_ = grid.i_phys(i, MPI_COMM_WORLD);
      const azeban::real_t expected_0
          = n * n * zisa::cos(2.0 * zisa::pi * (i_ + j) / n);
      const azeban::real_t expected_1
          = n * n * zisa::cos(4.0 * zisa::pi * (i_ + j) / n);
      REQUIRE(std::fabs(h_u(0, i, j) - expected_0) <= 1e-8);
      REQUIRE(std::fabs(h_u(1, i, j) - expected_1) <= 1e-8);
    }
  }
}

TEST_CASE("cuFFT MPI 3d scalar valued data", "[mpi]") {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  zisa::int_t n = 128;
  azeban::Grid<3> grid(n);
  zisa::shape_t<4> rshape = grid.shape_phys(1, MPI_COMM_WORLD);
  ;
  zisa::shape_t<4> cshape = grid.shape_fourier(1, MPI_COMM_WORLD);
  auto h_u = zisa::array<azeban::real_t, 4>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 4>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 4>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 4>(cshape);

  std::shared_ptr<azeban::FFT<3>> fft
      = std::make_shared<azeban::CUFFT_MPI<3>>(d_u_hat, d_u, MPI_COMM_WORLD);

  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    const zisa::int_t i_ = grid.i_phys(i, MPI_COMM_WORLD);
    for (zisa::int_t j = 0; j < rshape[2]; ++j) {
      for (zisa::int_t k = 0; k < rshape[3]; ++k) {
        h_u(0, i, j, k) = zisa::cos(2.0 * zisa::pi * (i_ + j + k) / n);
      }
    }
  }

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    for (zisa::int_t j = 0; j < cshape[2]; ++j) {
      for (zisa::int_t k = 0; k < cshape[3]; ++k) {
        const zisa::int_t i__ = grid.i_fourier(i, MPI_COMM_WORLD);
        const int i_ = i__ >= n / 2 + 1 ? zisa::integer_cast<int>(i__) - n
                                        : zisa::integer_cast<int>(i__);
        const int j_ = j >= n / 2 + 1 ? zisa::integer_cast<int>(j) - n
                                      : zisa::integer_cast<int>(j);
        const int k_ = k >= n / 2 + 1 ? zisa::integer_cast<int>(k) - n
                                      : zisa::integer_cast<int>(k);
        azeban::complex_t expected;
        expected.x = (i_ == 1 && j_ == 1 && k_ == 1)
                             || (i_ == -1 && j_ == -1 && k_ == -1)
                         ? n * n * n / 2.0
                         : 0.0;
        expected.y = 0;
        REQUIRE(std::fabs(h_u_hat(0, i, j, k).x - expected.x) <= 1e-8);
        REQUIRE(std::fabs(h_u_hat(0, i, j, k).y - expected.y) <= 1e-8);
      }
    }
  }

  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    for (zisa::int_t j = 0; j < rshape[2]; ++j) {
      for (zisa::int_t k = 0; k < rshape[3]; ++k) {
        const zisa::int_t i_ = grid.i_phys(i, MPI_COMM_WORLD);
        const azeban::real_t expected
            = n * n * n * zisa::cos(2.0 * zisa::pi * (i_ + j + k) / n);
        REQUIRE(std::fabs(h_u(0, i, j, k) - expected) <= 1e-8);
      }
    }
  }
}

TEST_CASE("cuFFT MPI 3d vector valued data", "[mpi]") {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  zisa::int_t n = 128;
  azeban::Grid<3> grid(n);
  zisa::shape_t<4> rshape = grid.shape_phys(3, MPI_COMM_WORLD);
  zisa::shape_t<4> cshape = grid.shape_fourier(3, MPI_COMM_WORLD);
  auto h_u = zisa::array<azeban::real_t, 4>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 4>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 4>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 4>(cshape);

  std::shared_ptr<azeban::FFT<3>> fft
      = std::make_shared<azeban::CUFFT_MPI<3>>(d_u_hat, d_u, MPI_COMM_WORLD);

  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    const zisa::int_t i_ = grid.i_phys(i, MPI_COMM_WORLD);
    for (zisa::int_t j = 0; j < rshape[2]; ++j) {
      for (zisa::int_t k = 0; k < rshape[3]; ++k) {
        h_u(0, i, j, k) = zisa::cos(2.0 * zisa::pi * (i_ + j + k) / n);
        h_u(1, i, j, k) = zisa::cos(4.0 * zisa::pi * (i_ + j + k) / n);
        h_u(2, i, j, k) = zisa::cos(6.0 * zisa::pi * (i_ + j + k) / n);
      }
    }
  }

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    for (zisa::int_t j = 0; j < cshape[2]; ++j) {
      for (zisa::int_t k = 0; k < cshape[3]; ++k) {
        const zisa::int_t i__ = grid.i_fourier(i, MPI_COMM_WORLD);
        const int i_ = i__ >= n / 2 + 1 ? zisa::integer_cast<int>(i__) - n
                                        : zisa::integer_cast<int>(i__);
        const int j_ = j >= n / 2 + 1 ? zisa::integer_cast<int>(j) - n
                                      : zisa::integer_cast<int>(j);
        const int k_ = k >= n / 2 + 1 ? zisa::integer_cast<int>(k) - n
                                      : zisa::integer_cast<int>(k);
        azeban::complex_t expected_0;
        expected_0.x = (i_ == 1 && j_ == 1 && k_ == 1)
                               || (i_ == -1 && j_ == -1 && k_ == -1)
                           ? n * n * n / 2.0
                           : 0.0;
        expected_0.y = 0;
        azeban::complex_t expected_1;
        expected_1.x = (i_ == 2 && j_ == 2 && k_ == 2)
                               || (i_ == -2 && j_ == -2 && k_ == -2)
                           ? n * n * n / 2.0
                           : 0.0;
        expected_1.y = 0;
        azeban::complex_t expected_2;
        expected_2.x = (i_ == 3 && j_ == 3 && k_ == 3)
                               || (i_ == -3 && j_ == -3 && k_ == -3)
                           ? n * n * n / 2.0
                           : 0.0;
        expected_2.y = 0;
        REQUIRE(std::fabs(h_u_hat(0, i, j, k).x - expected_0.x) <= 1e-8);
        REQUIRE(std::fabs(h_u_hat(0, i, j, k).y - expected_0.y) <= 1e-8);
        REQUIRE(std::fabs(h_u_hat(1, i, j, k).x - expected_1.x) <= 1e-8);
        REQUIRE(std::fabs(h_u_hat(1, i, j, k).y - expected_1.y) <= 1e-8);
        REQUIRE(std::fabs(h_u_hat(2, i, j, k).x - expected_2.x) <= 1e-8);
        REQUIRE(std::fabs(h_u_hat(2, i, j, k).y - expected_2.y) <= 1e-8);
      }
    }
  }

  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    for (zisa::int_t j = 0; j < rshape[2]; ++j) {
      for (zisa::int_t k = 0; k < rshape[3]; ++k) {
        const zisa::int_t i_ = grid.i_phys(i, MPI_COMM_WORLD);
        const azeban::real_t expected_0
            = n * n * n * zisa::cos(2.0 * zisa::pi * (i_ + j + k) / n);
        const azeban::real_t expected_1
            = n * n * n * zisa::cos(4.0 * zisa::pi * (i_ + j + k) / n);
        const azeban::real_t expected_2
            = n * n * n * zisa::cos(6.0 * zisa::pi * (i_ + j + k) / n);
        REQUIRE(std::fabs(h_u(0, i, j, k) - expected_0) <= 1e-8);
        REQUIRE(std::fabs(h_u(1, i, j, k) - expected_1) <= 1e-8);
        REQUIRE(std::fabs(h_u(2, i, j, k) - expected_2) <= 1e-8);
      }
    }
  }
}
#endif
