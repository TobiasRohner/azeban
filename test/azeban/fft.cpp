#include <azeban/catch.hpp>

#include <azeban/fft.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array_view.hpp>

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
  zisa::shape_t<2> rshape{1, n};
  zisa::shape_t<2> cshape{1, n / 2 + 1};
  auto h_u = zisa::array<azeban::real_t, 2>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 2>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 2>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 2>(cshape);

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

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    azeban::complex_t expected;
    expected.x = i == 1 ? n / 2.0 : 0.0;
    expected.y = 0;
    REQUIRE(std::fabs(h_u_hat[i].x - expected.x) <= 1e-10);
    REQUIRE(std::fabs(h_u_hat[i].y - expected.y) <= 1e-10);
  }
  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    azeban::real_t expected = n * zisa::cos(2.0 * zisa::pi * i / n);
    REQUIRE(std::fabs(h_u[i] - expected) <= 1e-10);
  }
}

TEST_CASE("cuFFT 1D vector valued data", "[cufft]") {
  zisa::int_t n = 128;
  zisa::shape_t<2> rshape{2, n};
  zisa::shape_t<2> cshape{2, n / 2 + 1};
  auto h_u = zisa::array<azeban::real_t, 2>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 2>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 2>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 2>(cshape);

  std::shared_ptr<azeban::FFT<1>> fft
      = std::make_shared<azeban::CUFFT<1>>(d_u_hat, d_u);

  for (zisa::int_t i = 0; i < n; ++i) {
    h_u(0, i) = zisa::cos(2.0 * zisa::pi * i / n);
    h_u(1, i) = zisa::cos(4.0 * zisa::pi * i / n);
  }

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    azeban::complex_t expected_0;
    expected_0.x = i == 1 ? n / 2.0 : 0.0;
    expected_0.y = 0;
    azeban::complex_t expected_1;
    expected_1.x = i == 2 ? n / 2.0 : 0.0;
    expected_1.y = 0;
    REQUIRE(std::fabs(h_u_hat(0, i).x - expected_0.x) <= 1e-10);
    REQUIRE(std::fabs(h_u_hat(0, i).y - expected_0.y) <= 1e-10);
    REQUIRE(std::fabs(h_u_hat(1, i).x - expected_1.x) <= 1e-10);
    REQUIRE(std::fabs(h_u_hat(1, i).y - expected_1.y) <= 1e-10);
  }
  for (zisa::int_t i = 0; i < rshape[1]; ++i) {
    azeban::real_t expected_0 = n * zisa::cos(2.0 * zisa::pi * i / n);
    REQUIRE(std::fabs(h_u(0, i) - expected_0) <= 1e-10);
    azeban::real_t expected_1 = n * zisa::cos(4.0 * zisa::pi * i / n);
    REQUIRE(std::fabs(h_u(1, i) - expected_1) <= 1e-10);
  }
}

TEST_CASE("cuFFT 2D scalar valued data", "[cufft]") {
  zisa::int_t n = 128;
  zisa::shape_t<3> rshape{1, n, n};
  zisa::shape_t<3> cshape{1, n, n / 2 + 1};
  auto h_u = zisa::array<azeban::real_t, 3>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 3>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 3>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 3>(cshape);

  std::shared_ptr<azeban::FFT<2>> fft
      = std::make_shared<azeban::CUFFT<2>>(d_u_hat, d_u);

  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      h_u(0, i, j) = zisa::cos(2.0 * zisa::pi * (i + j) / n);
    }
  }

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    for (zisa::int_t j = 0; j < cshape[2]; ++j) {
      const int i_ = i >= n / 2 + 1 ? zisa::integer_cast<int>(i) - n
                                    : zisa::integer_cast<int>(i);
      const int j_ = j >= n / 2 + 1 ? zisa::integer_cast<int>(j) - n
                                    : zisa::integer_cast<int>(j);
      azeban::complex_t expected;
      expected.x
          = (i_ == 1 && j_ == 1) || (i_ == -1 && j_ == -1) ? n * n / 2.0 : 0.0;
      expected.y = 0;
      REQUIRE(std::fabs(h_u_hat(0, i, j).x - expected.x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(0, i, j).y - expected.y) <= 1e-10);
    }
  }
}

TEST_CASE("cuFFT 2D vector valued data", "[cufft]") {
  zisa::int_t n = 128;
  zisa::shape_t<3> rshape{2, n, n};
  zisa::shape_t<3> cshape{2, n, n / 2 + 1};
  auto h_u = zisa::array<azeban::real_t, 3>(rshape);
  auto h_u_hat = zisa::array<azeban::complex_t, 3>(cshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 3>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 3>(cshape);

  std::shared_ptr<azeban::FFT<2>> fft
      = std::make_shared<azeban::CUFFT<2>>(d_u_hat, d_u);

  for (zisa::int_t i = 0; i < n; ++i) {
    for (zisa::int_t j = 0; j < n; ++j) {
      h_u(0, i, j) = zisa::cos(2.0 * zisa::pi * (i + j) / n);
      h_u(1, i, j) = zisa::cos(4.0 * zisa::pi * (i + j) / n);
    }
  }

  zisa::copy(d_u, h_u);
  fft->forward();
  zisa::copy(h_u_hat, d_u_hat);
  fft->backward();
  zisa::copy(h_u, d_u);

  for (zisa::int_t i = 0; i < cshape[1]; ++i) {
    for (zisa::int_t j = 0; j < cshape[2]; ++j) {
      const int i_ = i >= n / 2 + 1 ? zisa::integer_cast<int>(i) - n
                                    : zisa::integer_cast<int>(i);
      const int j_ = j >= n / 2 + 1 ? zisa::integer_cast<int>(j) - n
                                    : zisa::integer_cast<int>(j);
      azeban::complex_t expected_0;
      expected_0.x
          = (i_ == 1 && j_ == 1) || (i_ == -1 && j_ == -1) ? n * n / 2.0 : 0.0;
      expected_0.y = 0;
      azeban::complex_t expected_1;
      expected_1.x
          = (i_ == 2 && j_ == 2) || (i_ == -2 && j_ == -2) ? n * n / 2.0 : 0.0;
      expected_1.y = 0;
      REQUIRE(std::fabs(h_u_hat(0, i, j).x - expected_0.x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(0, i, j).x - expected_0.x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(1, i, j).y - expected_1.y) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(1, i, j).y - expected_1.y) <= 1e-10);
    }
  }
}
