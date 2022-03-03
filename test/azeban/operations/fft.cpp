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

#include <azeban/cuda/operations/cufft_mpi.hpp>
#include <azeban/grid.hpp>
#include <azeban/operations/fft_factory.hpp>
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

template <int Axis, int Dim, typename ScalarU>
static void dft_axis(const zisa::array_const_view<ScalarU, Dim> &in,
                     const zisa::array_view<azeban::complex_t, Dim> &out) {
  if constexpr (Axis == 0) {
    for (int i = 1; i < Dim; ++i) {
      if (in.shape(i) != out.shape(i)) {
        fmt::print("Error: Shape mismatch: in.shape({0}) = {1}, out.shape({0}) "
                   "= {2}\n",
                   i,
                   in.shape(i),
                   out.shape(i));
        LOG_ERR("Mismatching shapes");
      }
    }
    zisa::array<ScalarU, 1> in_buf(zisa::shape_t<1>(in.shape(0)),
                                   zisa::device_type::cpu);
    zisa::array<azeban::complex_t, 1> out_buf(zisa::shape_t<1>(out.shape(0)),
                                              zisa::device_type::cpu);
    const zisa::int_t stride = zisa::product(in.shape()) / in.shape(0);
    for (zisa::int_t i = 0; i < stride; ++i) {
      for (zisa::int_t j = 0; j < in.shape(0); ++j) {
        in_buf[j] = in[i + j * stride];
      }
      dft(in_buf, out_buf);
      for (zisa::int_t j = 0; j < out.shape(0); ++j) {
        out[i + j * stride] = out_buf[j];
      }
    }
  } else {
    zisa::shape_t<Dim - 1> in_slice_shape;
    zisa::shape_t<Dim - 1> out_slice_shape;
    for (int i = 0; i < Dim - 1; ++i) {
      in_slice_shape[i] = in.shape(i + 1);
      out_slice_shape[i] = out.shape(i + 1);
    }
    LOG_ERR_IF(in.shape(0) != out.shape(0), "Shape mismatch");
    for (zisa::int_t i = 0; i < in.shape(0); ++i) {
      zisa::array_const_view<ScalarU, Dim - 1> in_slice(
          in_slice_shape,
          in.raw() + i * zisa::product(in_slice_shape),
          zisa::device_type::cpu);
      zisa::array_view<azeban::complex_t, Dim - 1> out_slice(
          out_slice_shape,
          out.raw() + i * zisa::product(out_slice_shape),
          zisa::device_type::cpu);
      dft_axis<Axis - 1, Dim - 1, ScalarU>(in_slice, out_slice);
    }
  }
}

template <bool transform_x = true>
static void dft(const zisa::array_const_view<azeban::real_t, 1> &in,
                const zisa::array_view<azeban::complex_t, 1> &out) {
  dft_axis<0, 1, azeban::real_t>(in, out);
}

template <bool transform_x = true, bool transform_y = true>
static void dft(const zisa::array_const_view<azeban::real_t, 2> &in,
                const zisa::array_view<azeban::complex_t, 2> &out);

template <>
void dft<true, true>(const zisa::array_const_view<azeban::real_t, 2> &in,
                     const zisa::array_view<azeban::complex_t, 2> &out) {
  zisa::array<azeban::complex_t, 2> buf(out.shape(), zisa::device_type::cpu);
  dft_axis<1, 2, azeban::real_t>(in, buf);
  dft_axis<0, 2, azeban::complex_t>(buf, out);
}

template <>
void dft<true, false>(const zisa::array_const_view<azeban::real_t, 2> &in,
                      const zisa::array_view<azeban::complex_t, 2> &out) {
  dft_axis<0, 2, azeban::real_t>(in, out);
}

template <>
void dft<false, true>(const zisa::array_const_view<azeban::real_t, 2> &in,
                      const zisa::array_view<azeban::complex_t, 2> &out) {
  dft_axis<1, 2, azeban::real_t>(in, out);
}

template <bool transform_x = true,
          bool transform_y = true,
          bool transform_z = true>
static void dft(const zisa::array_const_view<azeban::real_t, 3> &in,
                const zisa::array_view<azeban::complex_t, 3> &out) {
  zisa::array<azeban::complex_t, 3> buf1(out.shape(), zisa::device_type::cpu);
  zisa::array<azeban::complex_t, 3> buf2(out.shape(), zisa::device_type::cpu);
  dft_axis<2, 3, azeban::real_t>(in, buf1);
  dft_axis<1, 3, azeban::complex_t>(buf1, buf2);
  dft_axis<0, 3, azeban::complex_t>(buf2, out);
}

template <>
void dft<true, true, true>(const zisa::array_const_view<azeban::real_t, 3> &in,
                           const zisa::array_view<azeban::complex_t, 3> &out) {
  zisa::array<azeban::complex_t, 3> buf1(out.shape(), zisa::device_type::cpu);
  zisa::array<azeban::complex_t, 3> buf2(out.shape(), zisa::device_type::cpu);
  dft_axis<2, 3, azeban::real_t>(in, buf1);
  dft_axis<1, 3, azeban::complex_t>(buf1, buf2);
  dft_axis<0, 3, azeban::complex_t>(buf2, out);
}

template <>
void dft<true, true, false>(const zisa::array_const_view<azeban::real_t, 3> &in,
                            const zisa::array_view<azeban::complex_t, 3> &out) {
  zisa::array<azeban::complex_t, 3> buf1(out.shape(), zisa::device_type::cpu);
  dft_axis<1, 3, azeban::real_t>(in, buf1);
  dft_axis<0, 3, azeban::complex_t>(buf1, out);
}

template <>
void dft<false, true, true>(const zisa::array_const_view<azeban::real_t, 3> &in,
                            const zisa::array_view<azeban::complex_t, 3> &out) {
  zisa::array<azeban::complex_t, 3> buf2(out.shape(), zisa::device_type::cpu);
  dft_axis<2, 3, azeban::real_t>(in, buf2);
  dft_axis<1, 3, azeban::complex_t>(buf2, out);
}

template <>
void dft<true, false, false>(
    const zisa::array_const_view<azeban::real_t, 3> &in,
    const zisa::array_view<azeban::complex_t, 3> &out) {
  dft_axis<0, 3, azeban::real_t>(in, out);
}
template <>

void dft<false, true, false>(
    const zisa::array_const_view<azeban::real_t, 3> &in,
    const zisa::array_view<azeban::complex_t, 3> &out) {
  dft_axis<1, 3, azeban::real_t>(in, out);
}

template <>
void dft<false, false, true>(
    const zisa::array_const_view<azeban::real_t, 3> &in,
    const zisa::array_view<azeban::complex_t, 3> &out) {
  dft_axis<2, 3, azeban::real_t>(in, out);
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
static void init(const zisa::array_view<azeban::complex_t, Dim> &u) {
  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  for (zisa::int_t i = 0; i < u.size(); ++i) {
    u[i].x = dist(rng);
    u[i].y = dist(rng);
  }
}

template <int Dim, bool... transform>
static void test_fft(zisa::int_t n,
                     zisa::int_t D,
                     zisa::device_type device = zisa::device_type::cpu) {
  static_assert(sizeof...(transform) == Dim, "");
  static constexpr int num_transformed_dims = (... + !!transform);

  const int direction = azeban::FFT_FORWARD | azeban::FFT_BACKWARD;
  std::shared_ptr<azeban::FFT<Dim>> fft
      = azeban::make_fft<Dim>(device, direction, transform...);

  zisa::shape_t<Dim + 1> rshape;
  rshape[0] = D;
  for (int i = 0; i < Dim; ++i) {
    rshape[i + 1] = n;
  }
  const zisa::shape_t<Dim + 1> cshape = fft->shape_u_hat(rshape);

  auto h_u = zisa::array<azeban::real_t, Dim + 1>(rshape);
  auto h_u_ref = zisa::array<azeban::real_t, Dim + 1>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, Dim + 1>(cshape);
  auto h_u_hat_ref = zisa::array<azeban::complex_t, Dim + 1>(cshape);
  auto d_u = zisa::array<azeban::real_t, Dim + 1>(rshape, device);
  auto d_u_hat = zisa::array<azeban::complex_t, Dim + 1>(cshape, device);

  fft->initialize(d_u_hat, d_u);

  zisa::shape_t<Dim> u_ref_view_shape;
  zisa::shape_t<Dim> u_hat_ref_view_shape;
  for (int i = 0; i < Dim; ++i) {
    u_ref_view_shape[i] = rshape[i + 1];
    u_hat_ref_view_shape[i] = cshape[i + 1];
  }
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
    dft<transform...>(u_ref_view, u_hat_ref_view);
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
    REQUIRE(std::fabs(h_u[i] / zisa::pow<num_transformed_dims>(n) - h_u_ref[i])
            <= 1e-8);
  }
}

#define REGISTER_1D_TEST_CASE(NVARS, TRANSX, DEV)                              \
  TEST_CASE("FFT 1D n_vars=" #NVARS " transform=(" #TRANSX "), device=" #DEV,  \
            "[operations][fft]") {                                             \
    std::cout << "TESTING: FFT 1D n_vars=" #NVARS " axes=(" #TRANSX            \
                 "), device=" #DEV " [operations][fft]"                        \
              << std::endl;                                                    \
    test_fft<1, TRANSX>(128, NVARS, DEV);                                      \
  }

#define REGISTER_2D_TEST_CASE(NVARS, TRANSX, TRANSY, DEV)                      \
  TEST_CASE("FFT 2D n_vars=" #NVARS " transform=(" #TRANSX ", " #TRANSY        \
            "), device=" #DEV,                                                 \
            "[operations][fft]") {                                             \
    std::cout << "TESTING: FFT 2D n_vars=" #NVARS " axes=(" #TRANSX            \
                 ", " #TRANSY "), device=" #DEV " [operations][fft]"           \
              << std::endl;                                                    \
    test_fft<2, TRANSX, TRANSY>(128, NVARS, DEV);                              \
  }

#define REGISTER_3D_TEST_CASE(NVARS, TRANSX, TRANSY, TRANSZ, DEV)              \
  TEST_CASE("FFT 3D n_vars=" #NVARS " transform=(" #TRANSX ", " #TRANSY        \
            ", " #TRANSZ "), device=" #DEV,                                    \
            "[operations][fft]") {                                             \
    std::cout << "TESTING: FFT 3D n_vars=" #NVARS " axes=(" #TRANSX            \
                 ", " #TRANSY ", " #TRANSZ "), device=" #DEV                   \
                 " [operations][fft]"                                          \
              << std::endl;                                                    \
    test_fft<3, TRANSX, TRANSY, TRANSZ>(128, NVARS, DEV);                      \
  }

REGISTER_1D_TEST_CASE(1, true, zisa::device_type::cpu);
REGISTER_1D_TEST_CASE(1, true, zisa::device_type::cuda);
REGISTER_1D_TEST_CASE(2, true, zisa::device_type::cpu);
REGISTER_1D_TEST_CASE(2, true, zisa::device_type::cuda);

REGISTER_2D_TEST_CASE(1, true, true, zisa::device_type::cpu);
REGISTER_2D_TEST_CASE(1, true, true, zisa::device_type::cuda);
REGISTER_2D_TEST_CASE(2, true, true, zisa::device_type::cpu);
REGISTER_2D_TEST_CASE(2, true, true, zisa::device_type::cuda);
// REGISTER_2D_TEST_CASE(1, true, false, zisa::device_type::cpu);
// REGISTER_2D_TEST_CASE(1, true, false, zisa::device_type::cuda);
// REGISTER_2D_TEST_CASE(2, true, false, zisa::device_type::cpu);
// REGISTER_2D_TEST_CASE(2, true, false, zisa::device_type::cuda);
REGISTER_2D_TEST_CASE(1, false, true, zisa::device_type::cpu);
REGISTER_2D_TEST_CASE(1, false, true, zisa::device_type::cuda);
REGISTER_2D_TEST_CASE(2, false, true, zisa::device_type::cpu);
REGISTER_2D_TEST_CASE(2, false, true, zisa::device_type::cuda);

REGISTER_3D_TEST_CASE(1, true, true, true, zisa::device_type::cpu);
REGISTER_3D_TEST_CASE(1, true, true, true, zisa::device_type::cuda);
REGISTER_3D_TEST_CASE(2, true, true, true, zisa::device_type::cpu);
REGISTER_3D_TEST_CASE(2, true, true, true, zisa::device_type::cuda);
// REGISTER_3D_TEST_CASE(1, true, true, false, zisa::device_type::cpu);
// REGISTER_3D_TEST_CASE(1, true, true, false, zisa::device_type::cuda);
// REGISTER_3D_TEST_CASE(2, true, true, false, zisa::device_type::cpu);
// REGISTER_3D_TEST_CASE(2, true, true, false, zisa::device_type::cuda);
REGISTER_3D_TEST_CASE(1, false, true, true, zisa::device_type::cpu);
REGISTER_3D_TEST_CASE(1, false, true, true, zisa::device_type::cuda);
REGISTER_3D_TEST_CASE(2, false, true, true, zisa::device_type::cpu);
REGISTER_3D_TEST_CASE(2, false, true, true, zisa::device_type::cuda);
// REGISTER_3D_TEST_CASE(1, true, false, false, zisa::device_type::cpu);
// REGISTER_3D_TEST_CASE(1, true, false, false, zisa::device_type::cuda);
// REGISTER_3D_TEST_CASE(2, true, false, false, zisa::device_type::cpu);
// REGISTER_3D_TEST_CASE(2, true, false, false, zisa::device_type::cuda);
REGISTER_3D_TEST_CASE(1, false, false, true, zisa::device_type::cpu);
REGISTER_3D_TEST_CASE(1, false, false, true, zisa::device_type::cuda);
REGISTER_3D_TEST_CASE(2, false, false, true, zisa::device_type::cpu);
REGISTER_3D_TEST_CASE(2, false, false, true, zisa::device_type::cuda);
// REGISTER_3D_TEST_CASE(1, false, true, false, zisa::device_type::cpu);
// REGISTER_3D_TEST_CASE(1, false, true, false, zisa::device_type::cuda);
// REGISTER_3D_TEST_CASE(2, false, true, false, zisa::device_type::cpu);
// REGISTER_3D_TEST_CASE(2, false, true, false, zisa::device_type::cuda);

#undef REGISTER_1D_TEST_CASE
#undef REGISTER_2D_TEST_CASE
#undef REGISTER_3D_TEST_CASE

TEST_CASE("cuFFT accuracy", "[operations][fft][this]") {
  std::cout << "TESTING: cuFFT accuracy [operations][fft]" << std::endl;
  azeban::Grid<2> grid(1024);
  auto h_u = grid.make_array_phys_pad(1, zisa::device_type::cpu);
  auto h_u_hat = grid.make_array_fourier_pad(1, zisa::device_type::cpu);
  auto d_u = grid.make_array_phys_pad(1, zisa::device_type::cuda);
  auto d_u_hat = grid.make_array_fourier_pad(1, zisa::device_type::cuda);
  auto fft = azeban::make_fft<2>(d_u_hat, d_u);
  zisa::fill(h_u_hat, azeban::complex_t(0));
  zisa::copy(d_u_hat, h_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);
  azeban::real_t val = 0;
  for (azeban::real_t u : h_u) {
    const azeban::real_t mag = std::fabs(u);
    if (mag > val) {
      val = mag;
    }
  }
  std::cout << "Maximum error is " << val << std::endl;
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
