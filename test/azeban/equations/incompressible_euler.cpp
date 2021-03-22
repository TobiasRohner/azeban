#include <azeban/catch.hpp>

#include <azeban/equations/incompressible_euler.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/evolution/ssp_rk2.hpp>
#include <azeban/evolution/ssp_rk3.hpp>
#include <azeban/init/discontinuous_vortex_patch.hpp>
#include <azeban/init/double_shear_layer.hpp>
#include <azeban/init/init_3d_from_2d.hpp>
#include <azeban/init/taylor_green.hpp>
#include <azeban/init/taylor_vortex.hpp>
#include <azeban/operations/fft.hpp>
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

template <int dim_v>
static azeban::real_t measureConvergence(
    const std::shared_ptr<azeban::Initializer<dim_v>> &initializer,
    zisa::int_t N_ref,
    azeban::real_t t) {
  const auto solve_euler
      = [&](const zisa::array_view<azeban::real_t, dim_v + 1> &u) {
          const zisa::int_t N = u.shape(1);
          azeban::Grid<dim_v> grid(N);
          azeban::SmoothCutoff1D visc(0.05 / N, 1);
          const auto equation = std::make_shared<
              azeban::IncompressibleEuler<dim_v, azeban::SmoothCutoff1D>>(
              grid, visc, zisa::device_type::cuda);
          const auto timestepper = std::make_shared<azeban::SSP_RK3<dim_v>>(
              zisa::device_type::cuda,
              grid.shape_fourier(equation->n_vars()),
              equation);
          azeban::CFL<dim_v> cfl(grid, 0.2);
          azeban::Simulation<dim_v> simulation(
              grid.shape_fourier(equation->n_vars()),
              cfl,
              timestepper,
              zisa::device_type::cuda);

          auto d_u = zisa::cuda_array<azeban::real_t, dim_v + 1>(
              grid.shape_phys(dim_v));
          const auto fft = azeban::make_fft<dim_v>(simulation.u(), d_u);

          initializer->initialize(simulation.u());
          simulation.simulate_until(t);
          fft->backward();
          zisa::copy(u, d_u);
          for (zisa::int_t i = 0; i < zisa::product(u.shape()); ++i) {
            u[i] /= zisa::product(u.shape()) / u.shape(0);
          }
        };

  zisa::shape_t<dim_v + 1> shape_ref;
  shape_ref[0] = dim_v;
  for (zisa::int_t i = 1; i <= dim_v; ++i) {
    shape_ref[i] = N_ref;
  }
  auto u_ref = zisa::array<azeban::real_t, dim_v + 1>(shape_ref);
  solve_euler(u_ref);

  std::vector<zisa::int_t> Ns;
  std::vector<azeban::real_t> errs;
  for (zisa::int_t N = 16; N < N_ref; N <<= 1) {
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = dim_v;
    for (zisa::int_t i = 1; i <= dim_v; ++i) {
      shape[i] = N;
    }
    auto u = zisa::array<azeban::real_t, dim_v + 1>(shape);
    solve_euler(u);
    azeban::real_t errL2 = 0;
    if constexpr (dim_v == 2) {
      for (zisa::int_t i = 0; i < N; ++i) {
        for (zisa::int_t j = 0; j < N; ++j) {
          const zisa::int_t i_ref = i * N_ref / N;
          const zisa::int_t j_ref = j * N_ref / N;
          const azeban::real_t du = u(0, i, j) - u_ref(0, i_ref, j_ref);
          const azeban::real_t dv = u(1, i, j) - u_ref(1, i_ref, j_ref);
          errL2 += zisa::pow<2>(du) + zisa::pow<2>(dv);
        }
      }
      errL2 = zisa::sqrt(errL2) / (N * N);
    } else {
      for (zisa::int_t i = 0; i < N; ++i) {
        for (zisa::int_t j = 0; j < N; ++j) {
          for (zisa::int_t k = 0; k < N; ++k) {
            const zisa::int_t i_ref = i * N_ref / N;
            const zisa::int_t j_ref = j * N_ref / N;
            const zisa::int_t k_ref = k * N_ref / N;
            const azeban::real_t du
                = u(0, i, j, k) - u_ref(0, i_ref, j_ref, k_ref);
            const azeban::real_t dv
                = u(1, i, j, k) - u_ref(1, i_ref, j_ref, k_ref);
            const azeban::real_t dw
                = u(2, i, j, k) - u_ref(2, i_ref, j_ref, k_ref);
            errL2 += zisa::pow<2>(du) + zisa::pow<2>(dv) + zisa::pow<2>(dw);
          }
        }
      }
      errL2 = zisa::sqrt(errL2) / (N * N * N);
    }
    Ns.push_back(N);
    errs.push_back(errL2);
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

  return conv_rate;
}

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

TEST_CASE("Taylor Vortex 2D", "[slow]") {
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
    const auto timestepper = std::make_shared<azeban::SSP_RK3<2>>(
        zisa::device_type::cuda,
        grid.shape_fourier(equation->n_vars()),
        equation);
    azeban::CFL<2> cfl(grid, 0.2);
    azeban::Simulation<2> simulation(grid.shape_fourier(equation->n_vars()),
                                     cfl,
                                     timestepper,
                                     zisa::device_type::cuda);
    initializer->initialize(simulation.u());

    auto d_u = zisa::cuda_array<azeban::real_t, 3>(grid.shape_phys(2));
    auto h_u = zisa::array<azeban::real_t, 3>(grid.shape_phys(2));
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
        const azeban::real_t u_ref_interp = u_ref(0, i_ref, j_ref) / 16;
        const azeban::real_t v_ref_interp = u_ref(1, i_ref, j_ref) / 16;
        const azeban::real_t du = h_u(0, i, j) - u_ref_interp;
        const azeban::real_t dv = h_u(1, i, j) - v_ref_interp;
        errL2 += zisa::pow<2>(du) + zisa::pow<2>(dv);
      }
    }
    errL2 = zisa::sqrt(errL2) / (N * N);
    Ns.push_back(N);
    errs.push_back(errL2);
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

TEST_CASE("Taylor Vortex 3D", "[slow]") {
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

  const auto test = [&](int dim) {
    const auto init2d = std::make_shared<azeban::TaylorVortex>();
    const auto initializer
        = std::make_shared<azeban::Init3DFrom2D>(dim, init2d);
    std::vector<zisa::int_t> Ns;
    std::vector<azeban::real_t> errs;
    for (zisa::int_t N = 16; N <= 128; N <<= 1) {
      azeban::Grid<3> grid(N);
      azeban::SmoothCutoff1D visc(0.05 / N, 1);
      const auto equation = std::make_shared<
          azeban::IncompressibleEuler<3, azeban::SmoothCutoff1D>>(
          grid, visc, zisa::device_type::cuda);
      const auto timestepper = std::make_shared<azeban::SSP_RK3<3>>(
          zisa::device_type::cuda,
          grid.shape_fourier(equation->n_vars()),
          equation);
      azeban::CFL<3> cfl(grid, 0.2);
      azeban::Simulation<3> simulation(grid.shape_fourier(equation->n_vars()),
                                       cfl,
                                       timestepper,
                                       zisa::device_type::cuda);
      initializer->initialize(simulation.u());

      auto d_u = zisa::cuda_array<azeban::real_t, 4>(grid.shape_phys(3));
      auto h_u = zisa::array<azeban::real_t, 4>(grid.shape_phys(3));
      const auto fft = azeban::make_fft<3>(simulation.u(), d_u);

      simulation.simulate_until(0.01);
      fft->backward();
      zisa::copy(h_u, d_u);
      for (zisa::int_t i = 0; i < zisa::product(h_u.shape()); ++i) {
        h_u[i] /= zisa::product(h_u.shape()) / h_u.shape(0);
      }

      azeban::real_t errL2 = 0;
      for (zisa::int_t i = 0; i < N; ++i) {
        for (zisa::int_t j = 0; j < N; ++j) {
          for (zisa::int_t k = 0; k < N; ++k) {
            const zisa::int_t i2d = dim > 0 ? i : j;
            const zisa::int_t j2d = dim > 1 ? j : k;
            const zisa::int_t i_ref = i2d * N_ref / N;
            const zisa::int_t j_ref = j2d * N_ref / N;
            const azeban::real_t u_ref_interp = u_ref(0, i_ref, j_ref) / 16;
            const azeban::real_t v_ref_interp = u_ref(1, i_ref, j_ref) / 16;
            const azeban::real_t du = h_u(0, i, j, k) - u_ref_interp;
            const azeban::real_t dv = h_u(1, i, j, k) - v_ref_interp;
            const azeban::real_t dw = h_u(2, i, j, k) - v_ref_interp;
            errL2 += zisa::pow<2>(du) + zisa::pow<2>(dv) + zisa::pow<2>(dw);
          }
        }
      }
      errL2 = zisa::sqrt(errL2) / (N * N * N);
      Ns.push_back(N);
      errs.push_back(errL2);
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
  };

  test(0);
  test(1);
  test(2);
}

TEST_CASE("Double Shear Layer 2D", "[slow]") {
  const auto initializer
      = std::make_shared<azeban::DoubleShearLayer>(0.2, 0.05);
  const azeban::real_t conv_rate = measureConvergence<2>(initializer, 512, 1);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Double Shear Layer 3D const. x", "[slow]") {
  const auto initializer2d
      = std::make_shared<azeban::DoubleShearLayer>(0.2, 0.05);
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(0, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 1);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Double Shear Layer 3D const. y", "[slow]") {
  const auto initializer2d
      = std::make_shared<azeban::DoubleShearLayer>(0.2, 0.05);
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(1, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 1);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Double Shear Layer 3D const. z", "[slow]") {
  const auto initializer2d
      = std::make_shared<azeban::DoubleShearLayer>(0.2, 0.05);
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(2, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 1);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Discontinous Vortex Patch 2D", "[slow]") {
  const auto initializer = std::make_shared<azeban::DiscontinuousVortexPatch>();
  const azeban::real_t conv_rate = measureConvergence<2>(initializer, 512, 5);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Discontinous Vortex Patch 3D const x", "[slow]") {
  const auto initializer2d
      = std::make_shared<azeban::DiscontinuousVortexPatch>();
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(0, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 5);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Discontinous Vortex Patch 3D const y", "[slow]") {
  const auto initializer2d
      = std::make_shared<azeban::DiscontinuousVortexPatch>();
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(1, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 5);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Discontinous Vortex Patch 3D const z", "[slow]") {
  const auto initializer2d
      = std::make_shared<azeban::DiscontinuousVortexPatch>();
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(2, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 5);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Taylor Green 2D", "[slow]") {
  const auto initializer = std::make_shared<azeban::TaylorGreen<2>>();
  const azeban::real_t conv_rate = measureConvergence<2>(initializer, 512, 1);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Taylor Green 3D", "[slow]") {
  const auto initializer = std::make_shared<azeban::TaylorGreen<3>>();
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 1);
  REQUIRE(conv_rate >= 1);
}
