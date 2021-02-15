#include <azeban/catch.hpp>

#include <azeban/fft.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>

TEST_CASE("Learn the cufft API", "[cufft]") {

  auto shape = zisa::shape_t<3>{16, 16, 16};
  auto u_uhat = zisa::cuda_array<cufftComplex, 3>(shape);

  cufftHandle plan;
  auto status = cufftPlan3d(&plan, shape[0], shape[1], shape[2], CUFFT_C2C);

  REQUIRE(status == CUFFT_SUCCESS);

}