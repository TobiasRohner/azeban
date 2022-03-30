#include <azeban/catch.hpp>
#include <azeban/config.hpp>
#include <azeban/operations/norm.hpp>
#include <iostream>
#include <zisa/cuda/memory/cuda_array.hpp>

TEST_CASE("norm complex CUDA", "[operations][norm]") {
  std::cout << "TESTING: norm complex CUDA [operations][norm]" << std::endl;
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

  REQUIRE(std::fabs(d - 1000) <= 1e-10);
}
