#include <azeban/catch.hpp>
#include <azeban/init/double_shear_layer.hpp>
#include <azeban/init/shear_tube.hpp>
#include <azeban/operations/leray.hpp>
#include <azeban/random/delta.hpp>
#include <iostream>
#include <zisa/cuda/memory/cuda_array.hpp>

TEST_CASE("Leray 2D CPU", "[operations][leray]") {
  std::cout << "TESTING: Leray 2D CPU [operations][leray]" << std::endl;
  zisa::int_t N = 32;
  zisa::shape_t<3> rshape{2, N, N};
  zisa::shape_t<3> cshape{2, N, N / 2 + 1};
  zisa::array<azeban::real_t, 3> u(rshape);
  zisa::array<azeban::complex_t, 3> uhat(cshape);
  auto fft = azeban::make_fft<2>(uhat, u);

  azeban::RandomVariable<azeban::real_t> rho(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.2));
  azeban::RandomVariable<azeban::real_t> delta(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.05));
  azeban::DoubleShearLayer init(rho, delta);
  init.initialize(u);
  fft->forward();
  leray(uhat);
  fft->backward();
  for (zisa::int_t i = 0; i < N; ++i) {
    for (zisa::int_t j = 0; j < N; ++j) {
      const zisa::int_t im = (i + N - 1) % N;
      const zisa::int_t ip = (i + 1) % N;
      const zisa::int_t jm = (j + N - 1) % N;
      const zisa::int_t jp = (j + 1) % N;
      const azeban::real_t dudx = (u(0, ip, j) - u(0, im, j)) / N;
      const azeban::real_t dvdy = (u(1, i, jp) - u(1, i, jm)) / N;
      const azeban::real_t div = dudx + dvdy;
      REQUIRE(std::fabs(div) < 1e-10);
    }
  }
}

TEST_CASE("Leray 3D CPU", "[operations][leray]") {
  std::cout << "TESTING: Leray 3D CPU [operations][leray]" << std::endl;
  zisa::int_t N = 32;
  zisa::shape_t<4> rshape{3, N, N, N};
  zisa::shape_t<4> cshape{3, N, N, N / 2 + 1};
  zisa::array<azeban::real_t, 4> u(rshape);
  zisa::array<azeban::complex_t, 4> uhat(cshape);
  auto fft = azeban::make_fft<3>(uhat, u);

  azeban::RandomVariable<azeban::real_t> rho(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.2));
  azeban::RandomVariable<azeban::real_t> delta(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.05));
  azeban::ShearTube init(rho, delta);
  init.initialize(u);
  fft->forward();
  leray(uhat);
  fft->backward();
  for (zisa::int_t i = 0; i < N; ++i) {
    for (zisa::int_t j = 0; j < N; ++j) {
      for (zisa::int_t k = 0; k < N; ++k) {
        const zisa::int_t im = (i + N - 1) % N;
        const zisa::int_t ip = (i + 1) % N;
        const zisa::int_t jm = (j + N - 1) % N;
        const zisa::int_t jp = (j + 1) % N;
        const zisa::int_t km = (k + N - 1) % N;
        const zisa::int_t kp = (k + 1) % N;
        const azeban::real_t dudx = (u(0, ip, j, k) - u(0, im, j, k)) / (N * N);
        const azeban::real_t dvdy = (u(1, i, jp, k) - u(1, i, jm, k)) / (N * N);
        const azeban::real_t dwdz = (u(2, i, j, kp) - u(2, i, j, km)) / (N * N);
        const azeban::real_t div = dudx + dvdy + dwdz;
        REQUIRE(std::fabs(div) < 1e-10);
      }
    }
  }
}

TEST_CASE("Leray 2D CUDA", "[operations][leray]") {
  std::cout << "TESTING: Leray 2D GPU [operations][leray]" << std::endl;
  zisa::int_t N = 32;
  zisa::shape_t<3> rshape{2, N, N};
  zisa::shape_t<3> cshape{2, N, N / 2 + 1};
  zisa::array<azeban::real_t, 3> hu(rshape, zisa::device_type::cpu);
  zisa::array<azeban::real_t, 3> du(rshape, zisa::device_type::cuda);
  zisa::array<azeban::complex_t, 3> uhat(cshape, zisa::device_type::cuda);
  auto fft = azeban::make_fft<2>(uhat, du);

  azeban::RandomVariable<azeban::real_t> rho(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.2));
  azeban::RandomVariable<azeban::real_t> delta(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.05));
  azeban::DoubleShearLayer init(rho, delta);
  init.initialize(du);
  fft->forward();
  leray(uhat);
  fft->backward();
  zisa::copy(hu, du);
  for (zisa::int_t i = 0; i < N; ++i) {
    for (zisa::int_t j = 0; j < N; ++j) {
      const zisa::int_t im = (i + N - 1) % N;
      const zisa::int_t ip = (i + 1) % N;
      const zisa::int_t jm = (j + N - 1) % N;
      const zisa::int_t jp = (j + 1) % N;
      const azeban::real_t dudx = (hu(0, ip, j) - hu(0, im, j)) / N;
      const azeban::real_t dvdy = (hu(1, i, jp) - hu(1, i, jm)) / N;
      const azeban::real_t div = dudx + dvdy;
      REQUIRE(std::fabs(div) < 1e-10);
    }
  }
}

TEST_CASE("Leray 3D CUDA", "[operations][leray]") {
  std::cout << "TESTING: Leray 3D GPU [operations][leray]" << std::endl;
  zisa::int_t N = 32;
  zisa::shape_t<4> rshape{3, N, N, N};
  zisa::shape_t<4> cshape{3, N, N, N / 2 + 1};
  zisa::array<azeban::real_t, 4> hu(rshape, zisa::device_type::cpu);
  zisa::array<azeban::real_t, 4> du(rshape, zisa::device_type::cuda);
  zisa::array<azeban::complex_t, 4> uhat(cshape, zisa::device_type::cuda);
  auto fft = azeban::make_fft<3>(uhat, du);

  azeban::RandomVariable<azeban::real_t> rho(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.2));
  azeban::RandomVariable<azeban::real_t> delta(
      std::make_shared<azeban::Delta<azeban::real_t>>(0.05));
  azeban::ShearTube init(rho, delta);
  init.initialize(du);
  fft->forward();
  leray(uhat);
  fft->backward();
  zisa::copy(hu, du);
  for (zisa::int_t i = 0; i < N; ++i) {
    for (zisa::int_t j = 0; j < N; ++j) {
      for (zisa::int_t k = 0; k < N; ++k) {
        const zisa::int_t im = (i + N - 1) % N;
        const zisa::int_t ip = (i + 1) % N;
        const zisa::int_t jm = (j + N - 1) % N;
        const zisa::int_t jp = (j + 1) % N;
        const zisa::int_t km = (k + N - 1) % N;
        const zisa::int_t kp = (k + 1) % N;
        const azeban::real_t dudx
            = (hu(0, ip, j, k) - hu(0, im, j, k)) / (N * N);
        const azeban::real_t dvdy
            = (hu(1, i, jp, k) - hu(1, i, jm, k)) / (N * N);
        const azeban::real_t dwdz
            = (hu(2, i, j, kp) - hu(2, i, j, km)) / (N * N);
        const azeban::real_t div = dudx + dvdy + dwdz;
        REQUIRE(std::fabs(div) < 1e-10);
      }
    }
  }
}
