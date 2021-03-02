#include <azeban/catch.hpp>

#include <azeban/equations/burgers.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>
#include <zisa/memory/array.hpp>

TEST_CASE("Burgers CUDA", "[equations]") {
  zisa::int_t n = 128;
  zisa::shape_t<1> shape{n};
  auto h_u = zisa::array<azeban::complex_t, 1>(shape);
  auto h_dudt = zisa::array<azeban::complex_t, 1>(shape);
  auto d_dudt = zisa::cuda_array<azeban::complex_t, 1>(shape);

  azeban::Burgers<azeban::Step1D> burgers(
      2 * (n - 1), azeban::Step1D(1, 20), zisa::device_type::cuda);

  for (zisa::int_t i = 0; i < n; ++i) {
    // h_u[i] = zisa::cos(2*10*zisa::pi * i / n);
    h_u[i] = 0;
  }
  h_u[1] = 0.5;

  zisa::copy(d_dudt, h_u);
  burgers.dudt(d_dudt);
  zisa::copy(h_dudt, d_dudt);

  std::cout << "u = [" << h_u[0];
  for (zisa::int_t i = 1; i < n; ++i) {
    std::cout << ", " << h_u[i];
  }
  std::cout << "]" << std::endl;
  std::cout << "dudt = [" << h_dudt[0];
  for (zisa::int_t i = 1; i < n; ++i) {
    std::cout << ", " << h_dudt[i];
  }
  std::cout << "]" << std::endl;
}
