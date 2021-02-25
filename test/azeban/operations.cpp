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

  std::cout << "u_hat = [";
  for (zisa::int_t i = 0 ; i < cshape[0]-1 ; ++i) {
    std::cout << '(' << clamp_to_zero(h_u_hat[i].x) << ", "
		     << clamp_to_zero(h_u_hat[i].y) << "), ";
  }
  std::cout << '(' << clamp_to_zero(h_u_hat[cshape[0]-1].x) << ", "
		   << clamp_to_zero(h_u_hat[cshape[0]-1].y) << ")]" << std::endl;

  zisa::copy(d_u_hat, h_u_hat);
  azeban::convolve_freq_domain(fft.get(), zisa::array_view<azeban::complex_t, 1>(d_u_hat));
  zisa::copy(h_u_hat, d_u_hat);

  std::cout << "u_hat = [";
  for (zisa::int_t i = 0 ; i < cshape[0]-1 ; ++i) {
    std::cout << '(' << clamp_to_zero(h_u_hat[i].x) << ", "
		     << clamp_to_zero(h_u_hat[i].y) << "), ";
  }
  std::cout << '(' << clamp_to_zero(h_u_hat[cshape[0]-1].x) << ", "
		   << clamp_to_zero(h_u_hat[cshape[0]-1].y) << ")]" << std::endl;

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
