#include <azeban/catch.hpp>

#include <azeban/equations/burgers.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/evolution/ssp_rk2.hpp>
#include <azeban/fft.hpp>
#include <azeban/grid.hpp>
#include <azeban/operations/operations.hpp>
#include <azeban/simulation.hpp>
#include <vector>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>
#include <zisa/memory/array.hpp>

static void solveBurgers(const azeban::Grid<1> &grid,
                         const zisa::array_view<azeban::real_t, 2> &h_u,
                         azeban::real_t visc,
                         azeban::real_t t) {
  auto d_u
      = zisa::cuda_array<azeban::real_t, 2>(zisa::shape_t<2>(1, grid.N_phys));
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 2>(
      zisa::shape_t<2>(1, grid.N_fourier));

  auto fft
      = azeban::make_fft<1>(zisa::array_view<azeban::complex_t, 2>(d_u_hat),
                            zisa::array_view<azeban::real_t, 2>(d_u));

  zisa::copy(d_u, h_u);
  fft->forward();

  azeban::CFL<1> cfl(grid, 0.1);
  auto equation = std::make_shared<azeban::Burgers<azeban::SmoothCutoff1D>>(
      grid, azeban::SmoothCutoff1D(visc, 1), zisa::device_type::cuda);
  auto timestepper = std::make_shared<azeban::SSP_RK2<azeban::complex_t, 1>>(
      zisa::device_type::cuda, d_u_hat.shape(), equation);
  auto simulation = azeban::Simulation<azeban::complex_t, 1>(
      zisa::array_const_view<azeban::complex_t, 2>(d_u_hat), cfl, timestepper);

  simulation.simulate_until(t);

  zisa::copy(d_u_hat, simulation.u());
  fft->backward();
  zisa::copy(h_u, d_u);
  for (zisa::int_t i = 0; i < grid.N_phys; ++i) {
    h_u[i] /= grid.N_phys;
  }
}

TEST_CASE("Burgers Derivative") {
  azeban::Grid<1> grid(128);
  zisa::int_t N_phys = grid.N_phys;
  zisa::int_t N_fourier = grid.N_fourier;
  zisa::shape_t<2> shape{1, grid.N_fourier};
  auto h_u = zisa::array<azeban::complex_t, 2>(shape);
  auto h_dudt = zisa::array<azeban::complex_t, 2>(shape);
  auto d_dudt = zisa::cuda_array<azeban::complex_t, 2>(shape);

  azeban::Burgers<azeban::Step1D> burgers(
      grid, azeban::Step1D(0, 0), zisa::device_type::cuda);

  for (zisa::int_t i = 0; i < N_fourier; ++i) {
    h_u[i] = 0;
  }
  h_u[1] = 0.5 * N_phys;

  zisa::copy(d_dudt, h_u);
  burgers.dudt(d_dudt);
  zisa::copy(h_dudt, d_dudt);

  for (zisa::int_t i = 0; i < N_fourier; ++i) {
    const azeban::real_t expected_x = 0;
    const azeban::real_t expected_y = i == 2 ? -zisa::pi * N_phys / 2 : 0;
    REQUIRE(std::fabs(h_dudt[i].x - expected_x) <= 1e-10);
    REQUIRE(std::fabs(h_dudt[i].y - expected_y) <= 1e-10);
  }
}

TEST_CASE("Burgers Convergence") {
  const auto compute_error
      = [&](const zisa::array_const_view<azeban::real_t, 2> &u_ref,
            const zisa::array_const_view<azeban::real_t, 2> &u) {
          using zisa::abs;
          azeban::real_t errL1 = 0;
          const zisa::int_t delta = u_ref.shape(1) / u.shape(1);
          for (zisa::int_t i = 0; i < u.shape(1); ++i) {
            const zisa::int_t i_ref = i * delta;
            errL1 += zisa::pow(abs(u[i] - u_ref[i_ref]), 2);
          }
          return zisa::pow(errL1, 0.5) / u.shape(1);
        };

  const azeban::Grid<1> grid_max(4 * 1024);
  const zisa::int_t N_max = grid_max.N_phys;
  const azeban::real_t visc = 0;
  const azeban::real_t t_final = 0.1; // At t=0.125 a shock develops

  auto u_ref = zisa::array<azeban::real_t, 2>(zisa::shape_t<2>{1, N_max});
  for (zisa::int_t i = 0; i < N_max; ++i) {
    u_ref[i] = zisa::sin(2 * zisa::pi / N_max * i);
  }
  solveBurgers(grid_max, u_ref, visc, t_final);

  std::vector<zisa::int_t> n;
  std::vector<azeban::real_t> err;
  for (zisa::int_t N = 128; N < N_max; N <<= 1) {
    azeban::Grid<1> grid(N);
    auto u = zisa::array<azeban::real_t, 2>(zisa::shape_t<2>{1, N});
    for (zisa::int_t i = 0; i < N; ++i) {
      u[i] = zisa::sin(2 * zisa::pi / N * i);
    }
    solveBurgers(grid, u, visc, t_final);
    n.push_back(N);
    err.push_back(compute_error(u_ref, u));
  }

  std::cout << "L1 errors = [" << err[0];
  for (zisa::int_t i = 1; i < err.size(); ++i) {
    std::cout << ", " << err[i];
  }
  std::cout << "]" << std::endl;

  const azeban::real_t conv_rate
      = (zisa::log(err[0]) - zisa::log(err[err.size() - 1]))
        / zisa::log(n[n.size() - 1] / n[0]);
  std::cout << "Estimated convergence rate: " << conv_rate << std::endl;

  REQUIRE(conv_rate >= 1);
  REQUIRE(conv_rate <= 2);
}

TEST_CASE("Burgers Shock Speed") {
  const azeban::Grid<1> grid(1024);
  const zisa::int_t N = grid.N_phys;
  const azeban::real_t visc = 0.05 / N;
  const azeban::real_t t_final = 0.5;

  auto u = zisa::array<azeban::real_t, 2>(zisa::shape_t<2>{1, N});
  for (zisa::int_t i = 0; i < N; ++i) {
    u[i] = i < N / 4 ? 1 : 0;
  }

  solveBurgers(grid, u, visc, t_final);

  REQUIRE(u[2 * N / 4 + N / 100] <= 0.01);
  REQUIRE(u[2 * N / 4 - N / 100] >= 0.9);
}

TEST_CASE("Burgers Corrctness Shock Free") {
  const auto dudt = [&](const zisa::array_const_view<azeban::real_t, 1> u,
                        zisa::array_view<azeban::real_t, 1> out) {
    const zisa::int_t N = u.shape(0);
    const azeban::real_t dx = 1. / N;
    for (zisa::int_t i = 0; i < N; ++i) {
      azeban::real_t ul;
      azeban::real_t ur;
      if (i == 0) {
        ul = u[N - 1];
        ur = u[1];
      } else if (i == N - 1) {
        ul = u[N - 2];
        ur = u[0];
      } else {
        ul = u[i - 1];
        ur = u[i + 1];
      }
      const azeban::real_t fl = 0.5 * ul * ul;
      const azeban::real_t fr = 0.5 * ur * ur;
      out[i] = (fl - fr) / (2 * dx);
    }
  };

  const auto cfl = [&](const zisa::array_const_view<azeban::real_t, 1> u,
                       azeban::real_t C) {
    const zisa::int_t N = u.shape(0);
    const azeban::real_t dx = 1. / N;
    azeban::real_t u_max = 0;
    for (zisa::int_t i = 0; i < N; ++i) {
      if (zisa::abs(u[i]) > u_max) {
        u_max = zisa::abs(u[i]);
      }
    }
    return C * dx / u_max;
  };

  const auto ssp_rk2
      = [&](const zisa::array_view<azeban::real_t, 1> u, azeban::real_t dt) {
          const zisa::int_t N = u.shape(0);
          zisa::array<azeban::real_t, 1> us(u.shape());
          zisa::array<azeban::real_t, 1> uss(u.shape());
          zisa::array<azeban::real_t, 1> diff(u.shape());
          dudt(u, diff);
          for (zisa::int_t i = 0; i < N; ++i) {
            us[i] = u[i] + dt * diff[i];
          }
          dudt(us, diff);
          for (zisa::int_t i = 0; i < N; ++i) {
            uss[i] = us[i] + dt * diff[i];
          }
          for (zisa::int_t i = 0; i < N; ++i) {
            u[i] = (u[i] + uss[i]) / 2;
          }
        };

  const auto solve_fd
      = [&](const zisa::array_view<azeban::real_t, 1> u, azeban::real_t t) {
          azeban::real_t time = 0;
          azeban::real_t dt = cfl(u, 0.1);
          while (t - time > dt) {
            ssp_rk2(u, dt);
            time += dt;
            dt = cfl(u, 0.1);
          }
          ssp_rk2(u, t - time);
        };

  const azeban::Grid<1> grid(1024);
  const zisa::int_t N = grid.N_phys;
  const azeban::real_t t_final = 0.1;

  auto u_ref = zisa::array<azeban::real_t, 1>(zisa::shape_t<1>{N});
  auto u_spectral = zisa::array<azeban::real_t, 2>(zisa::shape_t<2>{1, N});
  for (zisa::int_t i = 0; i < N; ++i) {
    u_ref[i] = zisa::sin(2 * zisa::pi / N * i);
    u_spectral[i] = zisa::sin(2 * zisa::pi / N * i);
  }

  solve_fd(u_ref, t_final);
  solveBurgers(grid, u_spectral, 0, t_final);

  azeban::real_t err = 0;
  for (zisa::int_t i = 0; i < N; ++i) {
    err += zisa::pow(u_ref[i] - u_spectral[i], 2);
  }
  err = zisa::sqrt(err);
  err /= N;
  REQUIRE(err <= 1e-5);
}
