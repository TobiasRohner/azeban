#include <azeban/catch.hpp>

#include <azeban/equations/incompressible_euler.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/evolution/ssp_rk2.hpp>
#include <azeban/evolution/ssp_rk3.hpp>
#include <azeban/fft.hpp>
#include <azeban/init/taylor_vortex.hpp>
#include <azeban/operations/operations.hpp>
#include <azeban/simulation.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <vector>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/io/hdf5_serial_writer.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>
#include <zisa/memory/array.hpp>

TEST_CASE("2D Euler Compute B") {
  azeban::Grid<2> grid(4);
  const zisa::int_t N_phys = grid.N_phys;
  const zisa::int_t N_fourier = grid.N_fourier;

  auto h_u_hat = grid.make_array_fourier(2, zisa::device_type::cpu);
  auto d_u_hat = grid.make_array_fourier(2, zisa::device_type::cuda);
  ;
  auto d_u = grid.make_array_phys(2, zisa::device_type::cuda);
  auto d_B = grid.make_array_phys(4, zisa::device_type::cuda);
  auto d_B_hat = grid.make_array_fourier(4, zisa::device_type::cuda);
  auto h_B_hat = grid.make_array_fourier(4, zisa::device_type::cpu);

  auto fft_u = azeban::make_fft<2>(d_u_hat, d_u);
  auto fft_B = azeban::make_fft<2>(d_B_hat, d_B);

  for (zisa::int_t i = 0; i < N_phys; ++i) {
    for (zisa::int_t j = 0; j < N_fourier; ++j) {
      h_u_hat(0, i, j) = 0;
      h_u_hat(1, i, j) = 0;
    }
  }
  h_u_hat(0, 0, 1) = 0.5 * N_phys * N_phys;
  h_u_hat(1, 1, 0) = 0.5 * N_phys * N_phys;
  h_u_hat(1, N_fourier, 0) = 0.5 * N_phys * N_phys;
  zisa::copy(d_u_hat, h_u_hat);

  fft_u->backward();
  azeban::incompressible_euler_compute_B_cuda<2>(fft_B->u(), fft_u->u(), grid);
  fft_B->forward();
  zisa::copy(h_B_hat, d_B_hat);

  for (zisa::int_t i = 0; i < N_phys; ++i) {
    for (zisa::int_t j = 0; j < N_fourier; ++j) {
      const azeban::complex_t expected = i == 0 && j == 1 ? 8 : 0;
      REQUIRE(std::fabs(h_u_hat(0, i, j).x - expected.x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(0, i, j).y - expected.y) <= 1e-10);
    }
  }
  for (zisa::int_t i = 0; i < N_phys; ++i) {
    for (zisa::int_t j = 0; j < N_fourier; ++j) {
      const azeban::complex_t expected = (i == 1 || i == 3) && j == 0 ? 8 : 0;
      REQUIRE(std::fabs(h_u_hat(1, i, j).x - expected.x) <= 1e-10);
      REQUIRE(std::fabs(h_u_hat(1, i, j).y - expected.y) <= 1e-10);
    }
  }
}

TEST_CASE("2D Euler Derivative") {
  const azeban::Grid<2> grid(4);
  const zisa::int_t N_phys = grid.N_phys;
  const zisa::int_t N_fourier = grid.N_fourier;

  auto h_u_hat = zisa::array<azeban::complex_t, 3>(
      zisa::shape_t<3>(2, N_phys, N_fourier));
  auto h_dudt_hat = zisa::array<azeban::complex_t, 3>(
      zisa::shape_t<3>(2, N_phys, N_fourier));
  auto d_dudt_hat = zisa::cuda_array<azeban::complex_t, 3>(
      zisa::shape_t<3>(2, N_phys, N_fourier));

  azeban::IncompressibleEuler<2, azeban::Step1D> euler(
      grid, azeban::Step1D(0, 0), zisa::device_type::cuda);

  for (zisa::int_t i = 0; i < N_phys; ++i) {
    for (zisa::int_t j = 0; j < N_fourier; ++j) {
      h_u_hat(0, i, j) = 0;
      h_u_hat(1, i, j) = 0;
    }
  }
  h_u_hat(0, 0, 1) = 0.5 * N_phys * N_phys;
  h_u_hat(1, 1, 0) = 0.5 * N_phys * N_phys;
  h_u_hat(1, N_fourier, 0) = 0.5 * N_phys * N_phys;

  zisa::copy(d_dudt_hat, h_u_hat);
  euler.dudt(d_dudt_hat);
  zisa::copy(h_dudt_hat, d_dudt_hat);

  for (zisa::int_t dim = 0; dim < 2; ++dim) {
    std::cout << "u_hat_" << dim << std::endl;
    for (zisa::int_t i = 0; i < N_phys; ++i) {
      for (zisa::int_t j = 0; j < N_fourier; ++j) {
        std::cout << h_u_hat(dim, i, j) << "\t";
      }
      std::cout << std::endl;
    }
  }
  for (zisa::int_t dim = 0; dim < 2; ++dim) {
    std::cout << "dudt_hat_" << dim << std::endl;
    for (zisa::int_t i = 0; i < N_phys; ++i) {
      for (zisa::int_t j = 0; j < N_fourier; ++j) {
        std::cout << h_dudt_hat(dim, i, j) << "\t";
      }
      std::cout << std::endl;
    }
  }
}

TEST_CASE("Taylor Vortex") {
  // Lagrange Polynomials for interpolation
  const auto L0 = [](azeban::real_t x) {
    return (x - 1) * (x - 2) * (x - 3) / (0 - 1) / (0 - 2) / (0 - 3);
  };
  const auto L1 = [](azeban::real_t x) {
    return (x - 0) * (x - 2) * (x - 3) / (1 - 0) / (1 - 2) / (1 - 3);
  };
  const auto L2 = [](azeban::real_t x) {
    return (x - 0) * (x - 1) * (x - 3) / (2 - 0) / (2 - 1) / (2 - 3);
  };
  const auto L3 = [](azeban::real_t x) {
    return (x - 0) * (x - 1) * (x - 2) / (3 - 0) / (3 - 1) / (3 - 2);
  };

  // Load reference solution
  // TODO: Get std::filesystem to not segfault
  // std::filesystem::path ref_sol_file =
  // std::filesystem::path(__FILE__).remove_filename() /
  // "../../../resources/momentum8748_taylor_morinishi6_T0,01.h5";
  // fmt::print("Loading Reference Solution from \"{}\"\n",
  // ref_sol_file.string());
  zisa::HDF5SerialReader reader(
      "../resources/momentum8748_taylor_morinishi6_T0,01.h5");
  const auto u_ref = zisa::array<azeban::real_t, 3>::load(reader, "0.010000");
  const zisa::int_t N_ref = u_ref.shape(1);

  const auto initializer = std::make_shared<azeban::TaylorVortex>();
  std::vector<zisa::int_t> Ns;
  std::vector<azeban::real_t> errs;
  for (zisa::int_t N = 128; N < 2048; N <<= 1) {
    azeban::Grid<2> grid(N);
    azeban::SmoothCutoff1D visc(0.05 / N, 1);
    const auto equation = std::make_shared<
        azeban::IncompressibleEuler<2, azeban::SmoothCutoff1D>>(
        grid, visc, zisa::device_type::cuda);
    const auto timestepper
        = std::make_shared<azeban::SSP_RK3<azeban::complex_t, 2>>(
            zisa::device_type::cuda,
            grid.shape_fourier(equation->n_vars()),
            equation);
    azeban::CFL<2> cfl(grid, 0.2);
    azeban::Simulation<azeban::complex_t, 2> simulation(
        grid.shape_fourier(equation->n_vars()),
        cfl,
        timestepper,
        zisa::device_type::cuda);
    initializer->initialize(simulation.u());

    auto d_u = zisa::cuda_array<azeban::real_t, 3>(grid.shape_phys(2));
    auto h_u = zisa::array<azeban::real_t, 3>(grid.shape_phys(2));
    auto u_diff = zisa::array<azeban::real_t, 3>(grid.shape_phys(2));
    const auto fft = azeban::make_fft<2>(simulation.u(), d_u);

    simulation.simulate_until(0.01);
    fft->backward();
    zisa::copy(h_u, d_u);
    for (zisa::int_t i = 0; i < zisa::product(h_u.shape()); ++i) {
      h_u[i] /= zisa::product(h_u.shape()) / h_u.shape(0);
    }

    azeban::real_t errL2 = 0;
    for (zisa::int_t i = 0; i < N; ++i) {
      for (zisa::int_t j = 0; j < N; ++j) {
        const zisa::int_t i_ref = i * N_ref / N;
        const zisa::int_t j_ref = j * N_ref / N;
        const azeban::real_t i_offset
            = static_cast<azeban::real_t>(i) * N_ref / N - i_ref;
        const azeban::real_t j_offset
            = static_cast<azeban::real_t>(j) * N_ref / N - j_ref;
        /*
        const zisa::int_t i_ref_m2 = (i_ref + N_ref - 1) % N_ref;
        const zisa::int_t i_ref_m1 = i_ref;
        const zisa::int_t i_ref_p1 = (i_ref + 1) % N_ref;
        const zisa::int_t i_ref_p2 = (i_ref + 2) % N_ref;
        const zisa::int_t j_ref_m2 = (j_ref + N_ref - 1) % N_ref;
        const zisa::int_t j_ref_m1 = j_ref;
        const zisa::int_t j_ref_p1 = (j_ref + 1) % N_ref;
        const zisa::int_t j_ref_p2 = (j_ref + 2) % N_ref;
        const azeban::real_t u_ref_m2 = u_ref(0, i_ref_m2, j) / 16;
        const azeban::real_t u_ref_m1 = u_ref(0, i_ref_m1, j) / 16;
        const azeban::real_t u_ref_p1 = u_ref(0, i_ref_p1, j) / 16;
        const azeban::real_t u_ref_p2 = u_ref(0, i_ref_p2, j) / 16;
        const azeban::real_t v_ref_m2 = u_ref(0, i, j_ref_m2) / 16;
        const azeban::real_t v_ref_m1 = u_ref(0, i, j_ref_m1) / 16;
        const azeban::real_t v_ref_p1 = u_ref(0, i, j_ref_p1) / 16;
        const azeban::real_t v_ref_p2 = u_ref(0, i, j_ref_p2) / 16;
        */
        const azeban::real_t u_ref_interp
            = u_ref(0, i_ref, j_ref)
              / 16; // u_ref_m2 * L0(1 + i_offset) + u_ref_m1 * L1(1 + i_offset)
                    // + u_ref_p1 * L2(1 + i_offset) + u_ref_p2 * L3(1 +
                    // i_offset);
        const azeban::real_t v_ref_interp
            = u_ref(1, i_ref, j_ref)
              / 16; // v_ref_m2 * L0(1 + j_offset) + v_ref_m1 * L1(1 + j_offset)
                    // + v_ref_p1 * L2(1 + j_offset) + v_ref_p2 * L3(1 +
                    // j_offset);
        const azeban::real_t du = h_u(0, i, j) - u_ref_interp;
        const azeban::real_t dv = h_u(1, i, j) - v_ref_interp;
        errL2 += zisa::pow<2>(du) + zisa::pow<2>(dv);
        u_diff(0, i, j) = du;
        u_diff(1, i, j) = dv;
      }
    }
    errL2 = zisa::sqrt(errL2) / (N * N);
    Ns.push_back(N);
    errs.push_back(errL2);

    zisa::HDF5SerialWriter writer("result_" + std::to_string(N) + ".hdf5");
    zisa::save(writer, u_diff, "u_diff");
  }

  std::cout << "L2 errors = [" << errs[0];
  for (zisa::int_t i = 1; i < errs.size(); ++i) {
    std::cout << ", " << errs[i];
  }
  std::cout << "]" << std::endl;

  const azeban::real_t conv_rate
      = (zisa::log(errs[0]) - zisa::log(errs[errs.size() - 1]))
        / zisa::log(Ns[Ns.size() - 1] / Ns[0]);
  std::cout << "Estimated convergence rate: " << conv_rate << std::endl;

  REQUIRE(conv_rate >= 1);
}
