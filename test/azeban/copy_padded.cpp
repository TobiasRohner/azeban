#include <azeban/catch.hpp>

#include <azeban/config.hpp>
#include <azeban/copy_padded.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>



TEST_CASE("Copy to padded CUDA 1D") {
  zisa::int_t n = 128;
  zisa::int_t n_pad = 3./2 * n;
  zisa::shape_t<1> shape{n};
  zisa::shape_t<1> shape_pad{n_pad};
  auto h_u = zisa::array<azeban::real_t, 1>(shape);
  auto d_u = zisa::cuda_array<azeban::real_t, 1>(shape);
  auto h_u_pad = zisa::array<azeban::real_t, 1>(shape_pad);
  auto d_u_pad = zisa::cuda_array<azeban::real_t, 1>(shape_pad);

  for (zisa::int_t i = 0 ; i < n ; ++i) {
    h_u[i] = zisa::cos(2.0 * zisa::pi * i / n);
  }

  zisa::copy(d_u, h_u);
  azeban::copy_to_padded(zisa::array_view<azeban::real_t, 1>(d_u_pad),
			 zisa::array_const_view<azeban::real_t, 1>(d_u),
			 azeban::real_t(0));
  zisa::copy(h_u_pad, d_u_pad);

  for (zisa::int_t i = 0 ; i < n_pad ; ++i) {
    azeban::real_t expected = i < n ? h_u[i] : azeban::real_t(0);
    REQUIRE(std::fabs(h_u_pad[i] - expected) <= 1e-10);
  }
}


TEST_CASE("Copy from padded CUDA 1D") {
  zisa::int_t n = 128;
  zisa::int_t n_pad = 3./2 * n;
  zisa::shape_t<1> shape{n};
  zisa::shape_t<1> shape_pad{n_pad};
  auto h_u = zisa::array<azeban::real_t, 1>(shape);
  auto d_u = zisa::cuda_array<azeban::real_t, 1>(shape);
  auto h_u_pad = zisa::array<azeban::real_t, 1>(shape_pad);
  auto d_u_pad = zisa::cuda_array<azeban::real_t, 1>(shape_pad);

  for (zisa::int_t i = 0 ; i < n_pad ; ++i) {
    h_u_pad[i] = zisa::cos(2.0 * zisa::pi * i / n_pad);
  }

  zisa::copy(d_u_pad, h_u_pad);
  azeban::copy_from_padded(zisa::array_view<azeban::real_t, 1>(d_u),
			   zisa::array_const_view<azeban::real_t, 1>(d_u_pad));
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0 ; i < n ; ++i) {
    azeban::real_t expected = h_u_pad[i];
    REQUIRE(std::fabs(h_u_pad[i] - expected) <= 1e-10);
  }
}
