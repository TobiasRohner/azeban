#include <azeban/catch.hpp>

#include <azeban/operations/operations.hpp>
#include <azeban/fft.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>


TEST_CASE("Convolution in Fourier Domain", "[operations]") {
  const auto clamp_to_zero = [](azeban::real_t value) {
    return std::fabs(value) <= 1e-10 ? azeban::real_t(0) : value;
  };
  
  zisa::int_t n = 128;
  zisa::shape_t<1> rshape{n};
  zisa::shape_t<1> cshape{n/2+1};
  auto h_u_hat = zisa::array<azeban::complex_t, 1>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 1>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 1>(cshape);

  std::shared_ptr<azeban::FFT<1>> fft = std::make_shared<azeban::CUFFT<1>>(d_u_hat, d_u);

  for (zisa::int_t i = 0 ; i < cshape[0] ; ++i) {
    h_u_hat[i] = 0;
  }
  h_u_hat[1] = 1;
  h_u_hat[2] = 2;

  zisa::copy(d_u_hat, h_u_hat);
  azeban::convolve_freq_domain(fft.get(), zisa::array_view<azeban::complex_t, 1>(d_u_hat));
  zisa::copy(h_u_hat, d_u_hat);

  for (zisa::int_t i = 0 ; i < cshape[0] ; ++i) {
    azeban::real_t expected = 0;
    switch (i) {
      case 0:
	expected = 10;
	break;
      case 1:
	expected = 4;
	break;
      case 2:
	expected = 1;
	break;
      case 3:
	expected = 4;
	break;
      case 4:
	expected = 4;
	break;
      default:
	expected = 0;
	break;
    }
    REQUIRE(azeban::abs(h_u_hat[i] - expected) <= 1e-10);
  }
}


TEST_CASE("axpy", "[operations]") {
  zisa::int_t n = 128;
  zisa::shape_t<1> shape{n};
  auto h_x = zisa::array<azeban::real_t, 1>(shape);
  auto d_x = zisa::cuda_array<azeban::real_t, 1>(shape);
  auto h_y = zisa::array<azeban::real_t, 1>(shape);
  auto d_y = zisa::cuda_array<azeban::real_t, 1>(shape);

  for (zisa::int_t i = 0 ; i < n ; ++i) {
    h_x[i] = 2*i;
    h_y[i] = n - i;
  }

  zisa::copy(d_x, h_x);
  zisa::copy(d_y, h_y);
  azeban::axpy(0.5,
	       zisa::array_const_view<azeban::real_t, 1>(d_x),
	       zisa::array_view<azeban::real_t, 1>(d_y));
  zisa::copy(h_x, d_x);
  zisa::copy(h_y, d_y);

  for (zisa::int_t i = 0 ; i < n ; ++i) {
    azeban::real_t expected_x = 2*i;
    azeban::real_t expected_y = n;
    REQUIRE(std::fabs(h_x[i] - expected_x) <= 1e-10);
    REQUIRE(std::fabs(h_y[i] - expected_y) <= 1e-10);
  }
}


TEST_CASE("norm complex CUDA", "[operations]") {
  zisa::int_t n = 100*100;
  zisa::shape_t<1> shape{n};
  auto h_u = zisa::array<azeban::complex_t, 1>(shape);
  auto d_u = zisa::cuda_array<azeban::complex_t, 1>(shape);

  for (zisa::int_t i = 0 ; i < n ; ++i) {
    h_u[i] = 1;
  }

  zisa::copy(d_u, h_u);
  const azeban::real_t d = azeban::norm(zisa::array_const_view<azeban::complex_t, 1>(d_u), 2);

  std::cout << "norm = " << d << std::endl;
  REQUIRE(std::fabs(d - 100) <= 1e-10);
}
