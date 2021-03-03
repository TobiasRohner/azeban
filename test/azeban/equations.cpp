#include <azeban/catch.hpp>

#include <azeban/equations/burgers.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>
#include <zisa/memory/array.hpp>

TEST_CASE("Burgers CUDA", "[equations]") {
  zisa::int_t N_phys = 128;
  zisa::int_t N_fourier = N_phys / 2 + 1;
  zisa::shape_t<1> shape{N_fourier};
  auto h_u = zisa::array<azeban::complex_t, 1>(shape);
  auto h_dudt = zisa::array<azeban::complex_t, 1>(shape);
  auto d_dudt = zisa::cuda_array<azeban::complex_t, 1>(shape);

  azeban::Burgers<azeban::Step1D> burgers(
      N_phys, azeban::Step1D(0, 0), zisa::device_type::cuda);

  for (zisa::int_t i = 0; i < N_fourier; ++i) {
    h_u[i] = 0;
  }
  h_u[1] = 0.5 * N_phys;

  zisa::copy(d_dudt, h_u);
  burgers.dudt(d_dudt);
  zisa::copy(h_dudt, d_dudt);

  for (zisa::int_t i = 0 ; i < N_fourier ; ++i) {
    const azeban::real_t expected_x = 0;
    const azeban::real_t expected_y = i == 2 ? -zisa::pi * N_phys / 2 : 0;
    REQUIRE(std::fabs(h_dudt[i].x - expected_x) <= 1e-10);
    REQUIRE(std::fabs(h_dudt[i].y - expected_y) <= 1e-10);
  }
}
