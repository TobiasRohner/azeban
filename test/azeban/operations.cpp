#include <azeban/catch.hpp>

#include <azeban/init/double_shear_layer.hpp>
#include <azeban/init/shear_tube.hpp>
#include <azeban/operations/fft.hpp>
#include <azeban/operations/leray.hpp>
#include <azeban/operations/operations.hpp>
#include <azeban/random/delta.hpp>
#include <azeban/random/random_variable.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>
#include <zisa/memory/array.hpp>

TEST_CASE("axpy", "[operations]") {
  zisa::int_t n = 128;
  zisa::shape_t<1> shape{n};
  auto h_x = zisa::array<azeban::real_t, 1>(shape);
  auto d_x = zisa::cuda_array<azeban::real_t, 1>(shape);
  auto h_y = zisa::array<azeban::real_t, 1>(shape);
  auto d_y = zisa::cuda_array<azeban::real_t, 1>(shape);

  for (zisa::int_t i = 0; i < n; ++i) {
    h_x[i] = 2 * i;
    h_y[i] = n - i;
  }

  zisa::copy(d_x, h_x);
  zisa::copy(d_y, h_y);
  azeban::axpy(azeban::real_t(0.5),
               zisa::array_const_view<azeban::real_t, 1>(d_x),
               zisa::array_view<azeban::real_t, 1>(d_y));
  zisa::copy(h_x, d_x);
  zisa::copy(h_y, d_y);

  for (zisa::int_t i = 0; i < n; ++i) {
    azeban::real_t expected_x = 2 * i;
    azeban::real_t expected_y = n;
    REQUIRE(std::fabs(h_x[i] - expected_x) <= 1e-10);
    REQUIRE(std::fabs(h_y[i] - expected_y) <= 1e-10);
  }
}

TEST_CASE("norm complex CUDA", "[operations]") {
  zisa::int_t n = 1000 * 1000;
  zisa::shape_t<1> shape{n};
  auto h_u = zisa::array<azeban::complex_t, 1>(shape);
  auto d_u = zisa::cuda_array<azeban::complex_t, 1>(shape);

  for (zisa::int_t i = 0; i < n; ++i) {
    h_u[i] = 1;
  }

  zisa::copy(d_u, h_u);
  const azeban::real_t d
      = azeban::norm(zisa::array_const_view<azeban::complex_t, 1>(d_u), 2);

  std::cout << "norm = " << d << std::endl;
  REQUIRE(std::fabs(d - 1000) <= 1e-10);
}

TEST_CASE("Leray 2D CPU", "[operations]") {
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

TEST_CASE("Leray 3D CPU", "[operations]") {
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

TEST_CASE("Leray 2D CUDA", "[operations]") {
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
  zisa::copy(du, hu);
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

TEST_CASE("Leray 3D CUDA", "[operations]") {
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
  zisa::copy(du, hu);
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
