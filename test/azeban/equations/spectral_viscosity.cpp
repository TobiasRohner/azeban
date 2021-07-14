#include <azeban/catch.hpp>

#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/init/const_phys.hpp>
#include <azeban/init/const_fourier_tracer.hpp>
#include <azeban/init/velocity_and_tracer.hpp>
#include <azeban/random/delta.hpp>
#include <azeban/equations/incompressible_euler.hpp>
#include <azeban/evolution/cfl.hpp>
#include <azeban/evolution/ssp_rk3.hpp>
#include <azeban/simulation.hpp>
#include <fmt/core.h>


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


template<typename Visc>
static void verify_via_tracer(Visc visc, zisa::int_t N) {
  azeban::Grid<2> grid(N);
  azeban::CFL<2> cfl(grid, 0.2);
  auto equation = std::make_shared<azeban::IncompressibleEuler<2, Visc>>(grid, visc, zisa::device_type::cpu, true);
  auto timestepper = std::make_shared<azeban::SSP_RK3<2>>(zisa::device_type::cpu, grid.shape_fourier(3), equation);
  azeban::RandomVariable<azeban::real_t> u(std::make_shared<azeban::Delta<azeban::real_t>>(0));
  auto init_u = std::make_shared<azeban::ConstPhys<2>>(u, u);
  auto init_rho = std::make_shared<azeban::ConstFourierTracer<2>>(1);
  auto initializer = std::make_shared<azeban::VelocityAndTracer<2>>(init_u, init_rho);
  azeban::Simulation<2> simulation(grid.shape_fourier(3), cfl, timestepper);
  initializer->initialize(simulation.u());
  const azeban::real_t dt = 0.001 / N;
  for (azeban::real_t t = dt ; t < 0.01 ; t += dt) {
    simulation.simulate_until(t);
    for (zisa::int_t i = 0 ; i < simulation.u().shape(1) ; ++i) {
      for (zisa::int_t j = 0 ; j < simulation.u().shape(2) ; ++j) {
	long i_ = i;
	if (i_ >= zisa::integer_cast<long>(N / 2 + 1)) {
	  i_ -= N;
	}
	long j_ = j;
	if (j_ >= zisa::integer_cast<long>(N / 2 + 1)) {
	  j_ -= N;
	}
	const azeban::real_t k1 = 2 * zisa::pi * i_;
	const azeban::real_t k2 = 2 * zisa::pi * j_;
	const azeban::real_t absk2 = k1 * k1 + k2 * k2;
	const azeban::real_t nu = visc.eval(zisa::sqrt(absk2));
	const azeban::real_t rho_analytical = zisa::exp(nu * t);
	REQUIRE(azeban::abs2(rho_analytical - simulation.u()(2, i, j)) <= 1e-7);
      }
    }
  }
}



TEST_CASE("Step1D cutoff 0", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::Step1D visc(0.05, 0);
    verify(visc, N);
  }
}

TEST_CASE("Step1D cutoff sqrt(N)", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::Step1D visc(0.05, sqrt(N));
    verify(visc, N);
  }
}

TEST_CASE("Step1D cutoff N", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::Step1D visc(0.05, N);
    verify(visc, N);
  }
}

TEST_CASE("SmoothCutoff1D cutoff 1", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::SmoothCutoff1D visc(0.05, 1);
    verify(visc, N);
  }
}

TEST_CASE("SmoothCutoff1D cutoff sqrt(N)", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::SmoothCutoff1D visc(0.05, sqrt(N));
    verify(visc, N);
  }
}

TEST_CASE("SmoothCutoff1D cutoff N", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::SmoothCutoff1D visc(0.05, N);
    verify(visc, N);
  }
}

TEST_CASE("Quadratic", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 4096 ; N <<= 1) {
    azeban::Quadratic visc(0.05, N);
    verify(visc, N);
  }
}

TEST_CASE("Step1D cutoff 0 tracer", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 256 ; N <<= 1) {
    azeban::Step1D visc(0.05, 0);
    verify_via_tracer(visc, N);
  }
}

TEST_CASE("Step1D cutoff sqrt(N) tracer", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 256 ; N <<= 1) {
    azeban::Step1D visc(0.05, sqrt(N));
    verify_via_tracer(visc, N);
  }
}

TEST_CASE("Step1D cutoff N tracer", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 256 ; N <<= 1) {
    azeban::Step1D visc(0.05, N);
    verify_via_tracer(visc, N);
  }
}

TEST_CASE("SmoothCutoff1D cutoff 1 tracer", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 256 ; N <<= 1) {
    azeban::SmoothCutoff1D visc(0.05, 1);
    verify_via_tracer(visc, N);
  }
}

TEST_CASE("SmoothCutoff1D cutoff sqrt(N) tracer", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 256 ; N <<= 1) {
    azeban::SmoothCutoff1D visc(0.05, sqrt(N));
    verify_via_tracer(visc, N);
  }
}

TEST_CASE("SmoothCutoff1D cutoff N tracer", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 256 ; N <<= 1) {
    azeban::SmoothCutoff1D visc(0.05, N);
    verify_via_tracer(visc, N);
  }
}

TEST_CASE("Quadratic tracer", "[visc]") {
  for (zisa::int_t N = 16 ; N <= 256 ; N <<= 1) {
    azeban::Quadratic visc(0.05, N);
    verify_via_tracer(visc, N);
  }
}
