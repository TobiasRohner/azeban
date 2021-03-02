#include <azeban/catch.hpp>

#include <azeban/equations/burgers.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/evolution/ssp_rk2.hpp>
#include <azeban/fft.hpp>
#include <azeban/operations/operations.hpp>
#include <azeban/simulation.hpp>
#include <vector>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>
#include <zisa/memory/array.hpp>

TEST_CASE("Burgers Convergence") {
  const auto solve_burgers = [&](const zisa::array_view<azeban::real_t, 1> &h_u,
                                 azeban::real_t visc,
                                 azeban::real_t t) {
    auto d_u = zisa::cuda_array<azeban::real_t, 1>(h_u.shape());
    auto d_u_hat = zisa::cuda_array<azeban::complex_t, 1>(
        zisa::shape_t<1>(h_u.shape(0) / 2 + 1));

    auto fft
        = azeban::make_fft<1>(zisa::array_view<azeban::complex_t, 1>(d_u_hat),
                              zisa::array_view<azeban::real_t, 1>(d_u));

    zisa::copy(d_u, h_u);
    fft->forward();

    azeban::CFL cfl(0.1);
    auto equation = std::make_shared<azeban::Burgers<azeban::SmoothCutoff1D>>(
        h_u.shape(0), azeban::SmoothCutoff1D(visc, 1), zisa::device_type::cuda);
    auto timestepper = std::make_shared<azeban::SSP_RK2<azeban::complex_t, 1>>(
        zisa::device_type::cuda, d_u_hat.shape(), equation);
    auto simulation = azeban::Simulation<azeban::complex_t, 1>(
        zisa::array_const_view<azeban::complex_t, 1>(d_u_hat),
        cfl,
        timestepper);

    simulation.simulate_until(t);

    zisa::copy(d_u_hat, simulation.u());
    fft->backward();
    zisa::copy(h_u, d_u);
    for (zisa::int_t i = 0; i < h_u.shape(0); ++i) {
      h_u[i] /= h_u.shape(0);
    }
  };

  const auto compute_error
      = [&](const zisa::array_const_view<azeban::real_t, 1> &u_ref,
            const zisa::array_const_view<azeban::real_t, 1> &u) {
          using zisa::abs;
          azeban::real_t errL1 = 0;
          const zisa::int_t delta = u_ref.shape(0) / u.shape(0);
          for (zisa::int_t i = 0; i < u.shape(0); ++i) {
            const zisa::int_t i_ref = i * delta;
            errL1 += zisa::pow(abs(u[i] - u_ref[i_ref]), 2);
          }
          return zisa::pow(errL1, 0.5) / u.shape(0);
        };

  const zisa::int_t N_max = 8 * 1024;
  const azeban::real_t visc = 0;
  const azeban::real_t t_final = 0.1; // At t=0.125 a shock develops

  auto u_ref = zisa::array<azeban::real_t, 1>(zisa::shape_t<1>{N_max});
  for (zisa::int_t i = 0; i < N_max; ++i) {
    u_ref[i] = zisa::sin(2 * zisa::pi / N_max * i);
  }
  solve_burgers(u_ref, visc, t_final);

  std::vector<zisa::int_t> n;
  std::vector<azeban::real_t> err;
  for (zisa::int_t N = 128; N < N_max; N <<= 1) {
    auto u = zisa::array<azeban::real_t, 1>(zisa::shape_t<1>{N});
    for (zisa::int_t i = 0; i < N; ++i) {
      u[i] = zisa::sin(2 * zisa::pi / N * i);
    }
    solve_burgers(u, visc, t_final);
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
}
