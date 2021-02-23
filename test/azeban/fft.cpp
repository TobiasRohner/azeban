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