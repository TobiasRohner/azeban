#include <azeban/catch.hpp>
#include <azeban/operations/fft_factory.hpp>
#include <azeban/operations/inverse_curl.hpp>
#include <iostream>

TEST_CASE("inverse_curl", "[operations][inverse_curl]") {
  std::cout << "TESTING: inverse_curl [operations][inverse_curl]" << std::endl;
  const zisa::shape_t<3> shape_u(2, 512, 512);
  const zisa::shape_t<3> shape_u_hat(2, 512, 512 / 2 + 1);
  const zisa::shape_t<3> shape_omega(1, 512, 512);
  const zisa::shape_t<3> shape_omega_hat(1, 512, 512 / 2 + 1);
  const zisa::shape_t<2> shape_omega_hat_view(512, 512 / 2 + 1);
  zisa::array<azeban::real_t, 3> omega(shape_omega, zisa::device_type::cpu);
  zisa::array<azeban::complex_t, 3> omega_hat(shape_omega_hat,
                                              zisa::device_type::cpu);
  zisa::array_view<azeban::complex_t, 2> omega_hat_view(
      shape_omega_hat_view, omega_hat.raw(), zisa::device_type::cpu);
  zisa::array<azeban::real_t, 3> u(shape_u, zisa::device_type::cpu);
  zisa::array<azeban::complex_t, 3> u_hat(shape_u_hat, zisa::device_type::cpu);
  auto fft_omega = azeban::make_fft<2>(omega_hat, omega);
  auto fft_u = azeban::make_fft<2>(u_hat, u);
  for (zisa::int_t i = 0; i < 512; ++i) {
    for (zisa::int_t j = 0; j < 512; ++j) {
      const azeban::real_t x = static_cast<azeban::real_t>(i) / 512;
      const azeban::real_t y = static_cast<azeban::real_t>(j) / 512;
      omega(0, i, j) = 4 * zisa::pi * zisa::cos(2 * zisa::pi * x)
                       * zisa::cos(2 * zisa::pi * y);
    }
  }
  fft_omega->forward();
  azeban::inverse_curl(omega_hat_view, u_hat);
  fft_u->backward();
  for (zisa::int_t i = 0; i < 512; ++i) {
    for (zisa::int_t j = 0; j < 512; ++j) {
      const azeban::real_t x = static_cast<azeban::real_t>(i) / 512;
      const azeban::real_t y = static_cast<azeban::real_t>(j) / 512;
      const azeban::real_t ex = 512 * 512 * zisa::cos(2 * zisa::pi * x)
                                * zisa::sin(2 * zisa::pi * y);
      const azeban::real_t ey = -512 * 512 * zisa::sin(2 * zisa::pi * x)
                                * zisa::cos(2 * zisa::pi * y);
      REQUIRE(std::fabs(u(0, i, j) - ex) <= 1e-8);
      REQUIRE(std::fabs(u(1, i, j) - ey) <= 1e-8);
    }
  }
}
