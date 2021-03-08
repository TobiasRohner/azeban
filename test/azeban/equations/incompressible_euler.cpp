#include <azeban/catch.hpp>

#include <azeban/equations/incompressible_euler.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/evolution/ssp_rk2.hpp>
#include <azeban/fft.hpp>
#include <azeban/operations/operations.hpp>
#include <azeban/simulation.hpp>
#include <vector>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>
#include <zisa/memory/array.hpp>

TEST_CASE("2D Euler Compute B") {
  azeban::Grid<2> grid(4);
  const zisa::int_t N_phys = grid.N_phys;
  const zisa::int_t N_fourier = grid.N_fourier;

  auto h_u_hat = grid.make_array_fourier(2, zisa::device_type::cpu);
  auto d_u_hat = grid.make_array_fourier(2, zisa::device_type::cuda);
  ;
  auto d_u = grid.make_array_phys(2, zisa::device_type::cuda);
  auto d_B = grid.make_array_phys(4, zisa::device_type::cuda);
  auto d_B_hat = grid.make_array_fourier(4, zisa::device_type::cuda);
  auto h_B_hat = grid.make_array_fourier(4, zisa::device_type::cpu);

  auto fft_u = azeban::make_fft<2>(d_u_hat, d_u);
  auto fft_B = azeban::make_fft<2>(d_B_hat, d_B);

  for (zisa::int_t i = 0; i < N_phys; ++i) {
    for (zisa::int_t j = 0; j < N_fourier; ++j) {
      h_u_hat(0, i, j) = 0;
      h_u_hat(1, i, j) = 0;
    }
  }
  h_u_hat(0, 0, 1) = 0.5 * N_phys * N_phys;
  h_u_hat(1, 1, 0) = 0.5 * N_phys * N_phys;
  h_u_hat(1, N_fourier, 0) = 0.5 * N_phys * N_phys;
  zisa::copy(d_u_hat, h_u_hat);

  fft_u->backward();
  azeban::incompressible_euler_compute_B_cuda<2>(fft_B->u(), fft_u->u(), grid);
  fft_B->forward();
  zisa::copy(h_B_hat, d_B_hat);

  for (zisa::int_t i = 0; i < N_phys; ++i) {
    for (zisa::int_t j = 0; j < N_fourier; ++j) {
      const azeban::complex_t expected = i == 0 && j == 1 ? 8 : 0;
      REQUIRE(std::fabs(h_u_hat(0, i, j).x - expected.x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(0, i, j).y - expected.y) <= 1e-10);
    }
  }
  for (zisa::int_t i = 0; i < N_phys; ++i) {
    for (zisa::int_t j = 0; j < N_fourier; ++j) {
      const azeban::complex_t expected = (i == 1 || i == 3) && j == 0 ? 8 : 0;
      REQUIRE(std::fabs(h_u_hat(1, i, j).x - expected.x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(1, i, j).y - expected.y) <= 1e-10);
    }
  }
}

TEST_CASE("2D Euler Derivative") {
  const azeban::Grid<2> grid(4);
  const zisa::int_t N_phys = grid.N_phys;
  const zisa::int_t N_fourier = grid.N_fourier;

  auto h_u_hat = zisa::array<azeban::complex_t, 3>(
      zisa::shape_t<3>(2, N_phys, N_fourier));
  auto h_dudt_hat = zisa::array<azeban::complex_t, 3>(
      zisa::shape_t<3>(2, N_phys, N_fourier));
  auto d_dudt_hat = zisa::cuda_array<azeban::complex_t, 3>(
      zisa::shape_t<3>(2, N_phys, N_fourier));

  azeban::IncompressibleEuler<2, azeban::Step1D> euler(
      grid, azeban::Step1D(0, 0), zisa::device_type::cuda);

  for (zisa::int_t i = 0; i < N_phys; ++i) {
    for (zisa::int_t j = 0; j < N_fourier; ++j) {
      h_u_hat(0, i, j) = 0;
      h_u_hat(1, i, j) = 0;
    }
  }
  h_u_hat(0, 0, 1) = 0.5 * N_phys * N_phys;
  h_u_hat(1, 1, 0) = 0.5 * N_phys * N_phys;
  h_u_hat(1, N_fourier, 0) = 0.5 * N_phys * N_phys;

  zisa::copy(d_dudt_hat, h_u_hat);
  euler.dudt(d_dudt_hat);
  zisa::copy(h_dudt_hat, d_dudt_hat);

  for (zisa::int_t dim = 0; dim < 2; ++dim) {
    std::cout << "u_hat_" << dim << std::endl;
    for (zisa::int_t i = 0; i < N_phys; ++i) {
      for (zisa::int_t j = 0; j < N_fourier; ++j) {
        std::cout << h_u_hat(dim, i, j) << "\t";
      }
      std::cout << std::endl;
    }
  }
  for (zisa::int_t dim = 0; dim < 2; ++dim) {
    std::cout << "dudt_hat_" << dim << std::endl;
    for (zisa::int_t i = 0; i < N_phys; ++i) {
      for (zisa::int_t j = 0; j < N_fourier; ++j) {
        std::cout << h_dudt_hat(dim, i, j) << "\t";
      }
      std::cout << std::endl;
    }
  }
}
