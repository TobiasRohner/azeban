#include <azeban/catch.hpp>

#include "utils.hpp"
#include <azeban/grid.hpp>
#include <azeban/operations/fft.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <cmath>
#include <fmt/core.h>


TEST_CASE("L2 Normal Distribution 2D", "[utils]") {
  const zisa::int_t N_ref = 1024;
  const azeban::Grid<2> grid_ref(N_ref);
  zisa::array<azeban::complex_t, 3> u_ref_hat(grid_ref.shape_fourier(1));
  zisa::fill(u_ref_hat, azeban::complex_t(0));
  for (zisa::int_t N = 32 ; N < N_ref ; N <<= 1) {
    for (azeban::real_t xbar : {-0.25, 0., 0.25}) {
      for (azeban::real_t ybar : {-0.25, 0., 0.25}) {
	for (azeban::real_t sigmax : {0.1, 1., 10.}) {
	  for (azeban::real_t sigmay : {0.1, 1.}) {
	    azeban::Grid<2> grid(N);
	    zisa::array<azeban::real_t, 3> u(grid.shape_phys(1));
	    zisa::array<azeban::complex_t, 3> u_hat(grid.shape_fourier(1));
	    const auto fft = azeban::make_fft<2>(u_hat, u);
	    for (zisa::int_t i = 0 ; i < N ; ++i) {
	      for (zisa::int_t j = 0 ; j < N ; ++j) {
		const azeban::real_t x = static_cast<azeban::real_t>(i) / N - 0.5;
		const azeban::real_t y = static_cast<azeban::real_t>(j) / N - 0.5;
		u(0, i, j) = 1;
		u(0, i, j) *= zisa::exp(-zisa::pow<2>((x - xbar) / sigmax));
		u(0, i, j) *= zisa::exp(-zisa::pow<2>((y - ybar) / sigmay));
	      }
	    }
	    fft->forward();
	    const azeban::real_t norm = L2<2>(u_hat, u_ref_hat);
	    azeban::real_t exact = zisa::pi * sigmax * sigmay / 8;
	    exact *= std::erf(zisa::sqrt(2) / sigmax * (0.5 - xbar)) - std::erf(zisa::sqrt(2) / sigmax * (-0.5 - xbar));
	    exact *= std::erf(zisa::sqrt(2) / sigmay * (0.5 - ybar)) - std::erf(zisa::sqrt(2) / sigmay * (-0.5 - ybar));
	    exact = zisa::sqrt(exact);
	    fmt::print(stderr, "N={}, xbar=({}, {}), sigma=({}, {}), result={}, expected={}\n", N, xbar, ybar, sigmax, sigmay, norm, exact);
	    REQUIRE(std::fabs(norm - exact) <= 1e-5);
	  }
	}
      }
    }
  }
}
