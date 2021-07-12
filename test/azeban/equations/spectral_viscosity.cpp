#include <azeban/catch.hpp>

#include <azeban/equations/spectral_viscosity.hpp>


template<typename Visc>
static void verify(Visc visc, zisa::int_t N) {
  azeban::real_t last_v = visc.Qk(0);
  REQUIRE((last_v >= 0 && last_v <= 1));
  for (zisa::int_t i = 1 ; i < N ; ++i) {
    const azeban::real_t k = 2 * zisa::pi * i;
    const azeban::real_t v = visc.Qk(k);
    REQUIRE(v >= last_v);
    REQUIRE((v >= 0 && v <= 1));
    last_v = v;
  }
  std::cout << std::endl;
}



TEST_CASE("Step1D cutoff 0") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::Step1D visc(0.05, 0);
    verify(visc, N);
  }
}

TEST_CASE("Step1D cutoff sqrt(N)") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::Step1D visc(0.05, sqrt(N));
    verify(visc, N);
  }
}

TEST_CASE("Step1D cutoff N") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::Step1D visc(0.05, N);
    verify(visc, N);
  }
}

TEST_CASE("SmoothCutoff1D cutoff 1") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::SmoothCutoff1D visc(0.05, 1);
    verify(visc, N);
  }
}

TEST_CASE("SmoothCutoff1D cutoff sqrt(N)") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::SmoothCutoff1D visc(0.05, sqrt(N));
    verify(visc, N);
  }
}

TEST_CASE("SmoothCutoff1D cutoff N") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::SmoothCutoff1D visc(0.05, N);
    verify(visc, N);
  }
}

TEST_CASE("Quadratic") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::Quadratic visc(0.05, N);
    verify(visc, N);
  }
}
