#include <azeban/catch.hpp>

#include <azeban/fft.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array_view.hpp>

TEST_CASE("Learn the cufft API; cufftComplex", "[cufft]") {
  cufftComplex z;
  z.x = 2.0;  // the real part
  z.y = -1.0; // the imaginary part
}

TEST_CASE("Learn the cufft API", "[cufft]") {

  zisa::int_t n = 128;
  auto shape = zisa::shape_t<1>{n};
  auto u = zisa::array<cufftComplex, 1>(shape);
  auto uhat = zisa::array<cufftComplex, 1>(shape);

  for (zisa::int_t i = 0; i < n; ++i) {
    u[i].x = zisa::cos(2.0 * zisa::pi * i / n);
    u[i].y = 0.0;
  }

  cufftHandle plan;
  auto status = cufftPlan1d(&plan, shape[0], CUFFT_C2C, 1);
  REQUIRE(status == CUFFT_SUCCESS);

  auto d_u = zisa::cuda_array<cufftComplex, 1>(shape);
  auto d_uhat = zisa::cuda_array<cufftComplex, 1>(shape);

  zisa::copy(d_u, u);
  cufftExecC2C(plan, d_u.raw(), d_uhat.raw(), CUFFT_FORWARD);
  zisa::copy(uhat, d_uhat);

  auto d_u2 = zisa::cuda_array<cufftComplex, 1>(shape);
  cufftExecC2C(plan, d_uhat.raw(), d_u2.raw(), CUFFT_INVERSE);

  auto u2 = zisa::array<cufftComplex, 1>(shape);
  zisa::copy(u2, d_u2);
  std::cout << uhat[0].x << ", " << uhat[1].x << ", " << uhat[2].x << "\n";
  std::cout << u2[0].x / n << ", " << u2[1].x / n << ", " << u2[2].x / n
            << "\n";
  std::cout << u[0].x << ", " << u[1].x << ", " << u[2].x << "\n";

  std::cout << (u2[0].x / n) - u[0].x << ", " << (u2[1].x / n) - u[1].x << ", "
            << (u2[2].x / n) - u[2].x << ", " << (u2[3].x / n) - u[3].x << "\n";
}

/*
TEST_CASE("FFTW scalar valued data", "[fft]") {
  zisa::int_t n = 128;
  zisa::shape_t<1> shape{n};
  zisa::array<azeban::real_t, 1> u(shape);
  zisa::array<azeban::complex_t, 1> u_hat(shape);

  std::shared_ptr<azeban::FFT<1>> fft =
std::make_shared<azeban::FFTWFFT<1>>(u_hat, u);

  for (zisa::int_t i = 0 ; i < n ; ++i) {
    u[i] = zisa::cos(2.0 * zisa::pi * i / n);
  }

  fft->forward();
  fft->backward();

  for (zisa::int_t i = 0 ; i < n ; ++i) {
    azeban::real_t expected = zisa::cos(2.0 * zisa::pi * i / n);
    REQUIRE(std::fabs(u[i] - expected) <= 1e-10);
  }
  for (zisa::int_t i = 0 ; i < n ; ++i) {
    azeban::complex_t expected;
    expected[0] = i == 2 ? n : 0;
    expected[1] = 0;
    REQUIRE(std::fabs(u_hat[i][0] - expected[0]) <= 1e-10);
    REQUIRE(std::fabs(u_hat[i][1] - expected[1]) <= 1e-10);
  }
}
*/

TEST_CASE("cuFFT 1D scalar valued data", "[cufft]") {
  zisa::int_t n = 128;
  zisa::shape_t<1> rshape{n};
  zisa::shape_t<1> cshape{n / 2 + 1};
  auto h_u = zisa::array<azeban::real_t, 1>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 1>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 1>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 1>(cshape);

  std::shared_ptr<azeban::FFT<1>> fft
      = std::make_shared<azeban::CUFFT<1>>(d_u_hat, d_u);

  for (zisa::int_t i = 0; i < n; ++i) {
    h_u[i] = zisa::cos(2.0 * zisa::pi * i / n);
  }

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < cshape[0]; ++i) {
    azeban::complex_t expected;
    expected.x = i == 1 ? n / 2.0 : 0.0;
    expected.y = 0;
    REQUIRE(std::fabs(h_u_hat[i].x - expected.x) <= 1e-10);
    REQUIRE(std::fabs(h_u_hat[i].y - expected.y) <= 1e-10);
  }
  for (zisa::int_t i = 0; i < rshape[0]; ++i) {
    azeban::real_t expected = n * zisa::cos(2.0 * zisa::pi * i / n);
    REQUIRE(std::fabs(h_u[i] - expected) <= 1e-10);
  }
}
