#include <azeban/catch.hpp>

#include <azeban/fft.hpp>
#include <azeban/operations/operations.hpp>
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
  azeban::axpy(0.5,
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
  zisa::int_t n = 100 * 100;
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
  REQUIRE(std::fabs(d - 100) <= 1e-10);
}
