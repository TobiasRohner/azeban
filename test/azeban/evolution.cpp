#include <azeban/catch.hpp>

#include <azeban/evolution/forward_euler.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>


TEST_CASE("Forward Euler CUDA", "[evolution]") {
  zisa::int_t n = 128;
  zisa::shape_t<1> shape{n};
  auto h_u = zisa::array<azeban::real_t, 1>(shape);
  auto d_u = zisa::cuda_array<azeban::real_t, 1>(shape);
  auto h_dudt = zisa::array<azeban::real_t, 1>(shape);
  auto d_dudt = zisa::cuda_array<azeban::real_t, 1>(shape);

  azeban::ForwardEuler<azeban::real_t, 1> euler;

  for (zisa::int_t i = 0 ; i < n ; ++i) {
    h_u[i] = i;
    h_dudt[i] = n - i;
  }

  zisa::copy(d_u, h_u);
  zisa::copy(d_dudt, h_dudt);
  euler.integrate(1,
		  zisa::array_view<azeban::real_t, 1>(d_u),
		  zisa::array_const_view<azeban::real_t, 1>(d_dudt));
  zisa::copy(h_u, d_u);
  zisa::copy(h_dudt, d_dudt);

  for (zisa::int_t i = 0 ; i < n ; ++i) {
    azeban::real_t expected_u = n;
    azeban::real_t expected_dudt = n - i;
    REQUIRE(std::fabs(h_u[i] - expected_u) <= 1e-10);
    REQUIRE(std::fabs(h_dudt[i] - expected_dudt) <= 1e-10);
  }
}
