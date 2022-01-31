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

#include <azeban/cuda/operations/cufft.hpp>
#include <azeban/cuda/operations/cufft_mpi.hpp>
#include <azeban/grid.hpp>
#include <azeban/operations/fft_factory.hpp>
#include <azeban/operations/fftwfft.hpp>
#include <fmt/core.h>
#include <iostream>
#include <random>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array_view.hpp>

static void dft(const zisa::array_const_view<azeban::real_t, 1> &in,
                const zisa::array_view<azeban::complex_t, 1> &out) {
  const zisa::int_t n = in.size();
  for (zisa::int_t i = 0; i < n / 2 + 1; ++i) {
    out[i] = 0;
    for (zisa::int_t k = 0; k < n; ++k) {
      const azeban::real_t xi
          = -2 * zisa::pi * static_cast<azeban::real_t>(k * i) / n;
      out[i] += in[k] * azeban::complex_t(zisa::cos(xi), zisa::sin(xi));
    }
  }
}

static void dft(const zisa::array_const_view<azeban::complex_t, 1> &in,
                const zisa::array_view<azeban::complex_t, 1> &out) {
  const zisa::int_t n = in.size();
  for (zisa::int_t i = 0; i < n; ++i) {
    out[i] = 0;
    for (zisa::int_t k = 0; k < n; ++k) {
      const azeban::real_t xi
          = -2 * zisa::pi * static_cast<azeban::real_t>(k * i) / n;
      out[i] += in[k] * azeban::complex_t(zisa::cos(xi), zisa::sin(xi));
    }
  }
}

static void dft(const zisa::array_const_view<azeban::real_t, 2> &in,
                const zisa::array_view<azeban::complex_t, 2> &out) {
  const zisa::int_t Nx = in.shape(0);
  const zisa::int_t Ny = in.shape(1);
  zisa::array<azeban::complex_t, 2> tmp(out.shape(), zisa::device_type::cpu);
  zisa::array<azeban::complex_t, 1> x_buf_in(zisa::shape_t<1>(Nx),
                                             zisa::device_type::cpu);
  zisa::array<azeban::complex_t, 1> x_buf_out(zisa::shape_t<1>(Nx),
                                              zisa::device_type::cpu);
  for (zisa::int_t i = 0; i < Nx; ++i) {
    zisa::array_const_view<azeban::real_t, 1> y_buf_in(
        zisa::shape_t<1>(Ny), &in(i, 0), zisa::device_type::cpu);
    zisa::array_view<azeban::complex_t, 1> y_buf_out(
        zisa::shape_t<1>(Ny / 2 + 1), &tmp(i, 0), zisa::device_type::cpu);
    dft(y_buf_in, y_buf_out);
  }
  for (zisa::int_t i = 0; i < Ny / 2 + 1; ++i) {
    for (zisa::int_t j = 0; j < Nx; ++j) {
      x_buf_in[j] = tmp(j, i);
    }
    dft(x_buf_in, x_buf_out);
    for (zisa::int_t j = 0; j < Nx; ++j) {
      out(j, i) = x_buf_out[j];
    }
  }
}

static void dft(const zisa::array_const_view<azeban::real_t, 3> &in,
                const zisa::array_view<azeban::complex_t, 3> &out) {
  const zisa::int_t Nx = in.shape(0);
  const zisa::int_t Ny = in.shape(1);
  const zisa::int_t Nz = in.shape(2);
  zisa::array<azeban::complex_t, 3> tmp(out.shape(), zisa::device_type::cpu);
  zisa::array<azeban::complex_t, 1> x_buf_in(zisa::shape_t<1>(Nx),
                                             zisa::device_type::cpu);
  zisa::array<azeban::complex_t, 1> x_buf_out(zisa::shape_t<1>(Nx),
                                              zisa::device_type::cpu);
  for (zisa::int_t i = 0; i < Nx; ++i) {
    zisa::array_const_view<azeban::real_t, 2> yz_buf_in(
        zisa::shape_t<2>(Ny, Nz), &in(i, 0, 0), zisa::device_type::cpu);
    zisa::array_view<azeban::complex_t, 2> yz_buf_out(
        zisa::shape_t<2>(Ny, Nz / 2 + 1),
        &tmp(i, 0, 0),
        zisa::device_type::cpu);
    dft(yz_buf_in, yz_buf_out);
  }
  for (zisa::int_t i = 0; i < Ny; ++i) {
    for (zisa::int_t j = 0; j < Nz / 2 + 1; ++j) {
      for (zisa::int_t k = 0; k < Nx; ++k) {
        x_buf_in[k] = tmp(k, i, j);
      }
      dft(x_buf_in, x_buf_out);
      for (zisa::int_t k = 0; k < Nx; ++k) {
        out(k, i, j) = x_buf_out[k];
      }
    }
  }
}

template <int Dim>
static void init(const zisa::array_view<azeban::real_t, Dim> &u) {
  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  for (zisa::int_t i = 0; i < u.size(); ++i) {
    u[i] = dist(rng);
  }
}

template <int Dim>
static void test_fft(zisa::int_t n,
                     zisa::int_t D,
                     zisa::device_type device = zisa::device_type::cpu) {
  zisa::shape_t<Dim + 1> rshape;
  zisa::shape_t<Dim + 1> cshape;
  rshape[0] = D;
  cshape[0] = D;
  for (int i = 0; i < Dim; ++i) {
    rshape[i + 1] = n;
    cshape[i + 1] = n;
  }
  cshape[Dim] = n / 2 + 1;
  auto h_u = zisa::array<azeban::real_t, Dim + 1>(rshape);
  auto h_u_ref = zisa::array<azeban::real_t, Dim + 1>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, Dim + 1>(cshape);
  auto h_u_hat_ref = zisa::array<azeban::complex_t, Dim + 1>(cshape);
  auto d_u = zisa::array<azeban::real_t, Dim + 1>(rshape, device);
  auto d_u_hat = zisa::array<azeban::complex_t, Dim + 1>(cshape, device);

  std::shared_ptr<azeban::FFT<Dim>> fft = azeban::make_fft<Dim>(d_u_hat, d_u);

  zisa::shape_t<Dim> u_ref_view_shape;
  for (int i = 0; i < Dim; ++i) {
    u_ref_view_shape[i] = n;
  }
  zisa::shape_t<Dim> u_hat_ref_view_shape = u_ref_view_shape;
  u_hat_ref_view_shape[Dim - 1] = n / 2 + 1;
  for (zisa::int_t d = 0; d < D; ++d) {
    zisa::array_view<azeban::real_t, Dim> u_ref_view(
        u_ref_view_shape,
        h_u_ref.raw() + d * zisa::product(u_ref_view_shape),
        zisa::device_type::cpu);
    zisa::array_view<azeban::complex_t, Dim> u_hat_ref_view(
        u_hat_ref_view_shape,
        h_u_hat_ref.raw() + d * zisa::product(u_hat_ref_view_shape),
        zisa::device_type::cpu);
    init(u_ref_view);
    dft(u_ref_view, u_hat_ref_view);
  }
  zisa::copy(h_u, h_u_ref);

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < zisa::product(cshape); ++i) {
    REQUIRE(std::fabs(h_u_hat[i].x - h_u_hat_ref[i].x) <= 1e-8);
    REQUIRE(std::fabs(h_u_hat[i].y - h_u_hat_ref[i].y) <= 1e-8);
  }
  for (zisa::int_t i = 0; i < zisa::product(rshape); ++i) {
    REQUIRE(std::fabs(h_u[i] / zisa::pow<Dim>(n) - h_u_ref[i]) <= 1e-8);
  }
}

TEST_CASE("FFTW 1D scalar valued data", "[operations][fft]") {
  std::cout << "TESTING: FFTW 1D scalar valued data [operations][fft]"
            << std::endl;
  test_fft<1>(128, 1);
}

TEST_CASE("FFTW 1D vector valued data", "[operations][fft]") {
  std::cout << "TESTING: FFTW 1D vector valued data [operations][fft]"
            << std::endl;
  test_fft<1>(128, 2);
}

TEST_CASE("FFTW 2D scalar valued data", "[operations][fft]") {
  std::cout << "TESTING: FFTW 2D scalar valued data [operations][fft]"
            << std::endl;
  test_fft<2>(128, 1);
}

TEST_CASE("FFTW 2D vector valued data", "[operations][fft]") {
  std::cout << "TESTING: FFTW 2D vector valued data [operations][fft]"
            << std::endl;
  test_fft<2>(128, 2);
}

TEST_CASE("FFTW 3D scalar valued data", "[operations][fft]") {
  std::cout << "TESTING: FFTW 3D scalar valued data [operations][fft]"
            << std::endl;
  test_fft<3>(128, 1);
}

TEST_CASE("FFTW 3D vector valued data", "[operations][fft]") {
  std::cout << "TESTING: FFTW 3D vector valued data [operations][fft]"
            << std::endl;
  test_fft<3>(128, 2);
}

TEST_CASE("cuFFT 1D scalar valued data", "[operations][fft]") {
  std::cout << "TESTING: cuFFT 1D scalar valued data [operations][fft]"
            << std::endl;
  test_fft<1>(128, 1, zisa::device_type::cuda);
}

TEST_CASE("cuFFT 1D vector valued data", "[operations][fft]") {
  std::cout << "TESTING: cuFFT 1D vector valued data [operations][fft]"
            << std::endl;
  test_fft<1>(128, 2, zisa::device_type::cuda);
}

TEST_CASE("cuFFT 2D scalar valued data", "[operations][fft]") {
  std::cout << "TESTING: cuFFT 2D scalar valued data [operations][fft]"
            << std::endl;
  test_fft<2>(128, 1, zisa::device_type::cuda);
}

TEST_CASE("cuFFT 2D vector valued data", "[operations][fft]") {
  std::cout << "TESTING: cuFFT 2D vector valued data [operations][fft]"
            << std::endl;
  test_fft<2>(128, 2, zisa::device_type::cuda);
}

TEST_CASE("cuFFT 3D scalar valued data", "[operations][fft]") {
  std::cout << "TESTING: cuFFT 3D scalar valued data [operations][fft]"
            << std::endl;
  test_fft<3>(128, 1, zisa::device_type::cuda);
}

TEST_CASE("cuFFT 3D vector valued data", "[operations][fft]") {
  std::cout << "TESTING: cuFFT 3D vector valued data [operations][fft]"
            << std::endl;
  test_fft<3>(128, 2, zisa::device_type::cuda);
}

#if AZEBAN_HAS_MPI
TEST_CASE("cuFFT MPI 2d scalar valued data", "[operations][fft][mpi]") {
  std::cout << "TESTING: cuFFT MPI 2d scalar valued data [operations][fft][mpi]"
            << std::endl;
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

TEST_CASE("cuFFT MPI 2d vector valued data", "[operations][fft][mpi]") {
  std::cout << "TESTING: cuFFT MPI 2d vector valued data [operations][fft][mpi]"
            << std::endl;
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

TEST_CASE("cuFFT MPI 3d scalar valued data", "[operations][fft][mpi]") {
  std::cout << "TESTING: cuFFT MPI 3d scalar valued data [operations][fft][mpi]"
            << std::endl;
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

TEST_CASE("cuFFT MPI 3d vector valued data", "[operations][fft][mpi]") {
  std::cout << "TESTING: cuFFT MPI 3d vector valued data [operations][fft][mpi]"
            << std::endl;
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
