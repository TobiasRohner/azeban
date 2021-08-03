#include <azeban/catch.hpp>

#include "../utils.hpp"
#include <azeban/equations/incompressible_euler.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/evolution/ssp_rk2.hpp>
#include <azeban/evolution/ssp_rk3.hpp>
#include <azeban/init/discontinuous_vortex_patch.hpp>
#include <azeban/init/double_shear_layer.hpp>
#include <azeban/init/init_3d_from_2d.hpp>
#include <azeban/init/taylor_green.hpp>
#include <azeban/init/taylor_vortex.hpp>
#include <azeban/operations/copy_padded.hpp>
#include <azeban/operations/fft.hpp>
#include <azeban/operations/operations.hpp>
#include <azeban/random/delta.hpp>
#include <azeban/simulation.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <vector>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/io/netcdf_serial_writer.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/math/mathematical_constants.hpp>
#include <zisa/memory/array.hpp>

template <int dim_v>
static azeban::real_t measureConvergence(
    const std::shared_ptr<azeban::Initializer<dim_v>> &initializer,
    zisa::int_t N_ref,
    azeban::real_t t) {
  const auto solve_euler
      = [&](zisa::int_t N) {
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

          initializer->initialize(simulation.u());
          simulation.simulate_until(t);
	  auto h_u_hat = zisa::array<azeban::complex_t, dim_v + 1>(grid.shape_fourier(dim_v));
	  zisa::copy(h_u_hat, simulation.u());
	  return h_u_hat;
        };

  const auto u_ref_hat = solve_euler(N_ref);

  std::vector<zisa::int_t> Ns;
  std::vector<azeban::real_t> errs;
  for (zisa::int_t N = 16; N < N_ref; N <<= 1) {
    const auto u_hat = solve_euler(N);
    const azeban::real_t errL2 = L2<dim_v>(u_hat, u_ref_hat);
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

static zisa::array<azeban::real_t, 3> read_reference() {
  const std::string filename
      = "momentum8748_taylor_morinishi6_T0,01_reshaped.nc";
  int status, ncid, varid;
  int dimids[3];
  size_t dims[3];
  status = nc_open(filename.c_str(), 0, &ncid);
  LOG_ERR_IF(status != NC_NOERR, "Failed to open reference solution");
  status = nc_inq_varid(ncid, "0.010000", &varid);
  LOG_ERR_IF(status != NC_NOERR, "Failed to open dataset");
  status = nc_inq_vardimid(ncid, varid, dimids);
  LOG_ERR_IF(status != NC_NOERR, "Failed to get dimensions");
  for (int i = 0; i < 3; ++i) {
    status = nc_inq_dimlen(ncid, dimids[i], dims + i);
    LOG_ERR_IF(status != NC_NOERR, "Failed to get dimension length");
  }
  const zisa::shape_t<3> shape{dims[0], dims[1], dims[2]};
  zisa::array<azeban::real_t, 3> reference(shape);
  status = nc_get_var(ncid, varid, reference.raw());
  LOG_ERR_IF(status != NC_NOERR, "Failed to read reference data");
  return reference;
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
  const auto u_ref = read_reference();
  const zisa::int_t N_ref = u_ref.shape(1);

  auto u_pad_hat = zisa::array<azeban::complex_t, 3>({2, N_ref, N_ref / 2 + 1});
  auto u_pad = zisa::array<azeban::real_t, 3>({2, N_ref, N_ref});
  const auto fft = azeban::make_fft<2>(u_pad_hat, u_pad);

  const auto initializer = std::make_shared<azeban::TaylorVortex>();
  std::vector<zisa::int_t> Ns;
  std::vector<azeban::real_t> errs;
  for (zisa::int_t N = 16; N <= 128; N <<= 1) {
    azeban::Grid<2> grid(N);
    azeban::SmoothCutoff1D visc(0, 1);
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

    auto h_u_hat = zisa::array<azeban::complex_t, 3>(grid.shape_fourier(2));

    simulation.simulate_until(0.01);
    zisa::copy(h_u_hat, simulation.u());
    azeban::copy_to_padded(component(u_pad_hat, 0), component(h_u_hat, 0), 0);
    azeban::copy_to_padded(component(u_pad_hat, 1), component(h_u_hat, 1), 0);
    fft->backward();
    for (zisa::int_t i = 0; i < zisa::product(u_pad.shape()); ++i) {
      u_pad[i] /= zisa::product(u_pad.shape()) / u_pad.shape(0);
      u_pad[i] *= zisa::pow<2>(static_cast<azeban::real_t>(N_ref) / N);
    }

    azeban::real_t errL2 = 0;
    for (zisa::int_t i = 0; i < N_ref; ++i) {
      for (zisa::int_t j = 0; j < N_ref; ++j) {
        const azeban::real_t u_ref_interp = u_ref(0, i, j) / 16;
        const azeban::real_t v_ref_interp = u_ref(1, i, j) / 16;
        const azeban::real_t du = u_pad(0, i, j) - u_ref_interp;
        const azeban::real_t dv = u_pad(1, i, j) - v_ref_interp;
        const azeban::real_t err_loc = zisa::pow<2>(du) + zisa::pow<2>(dv);
        errL2 += err_loc;
      }
    }
    errL2 = zisa::sqrt(errL2) / (N_ref * N_ref);
    Ns.push_back(N);
    errs.push_back(errL2);
  }

  std::cout << "Taylor Vortex 2D L2 errors = [" << errs[0];
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
  const auto u_ref = read_reference();
  const zisa::int_t N_ref = u_ref.shape(1);

  auto u_pad_hat
      = zisa::array<azeban::complex_t, 4>({2, N_ref, N_ref, N_ref / 2 + 1});
  auto u_pad = zisa::array<azeban::real_t, 4>({2, N_ref, N_ref, N_ref});
  const auto fft = azeban::make_fft<3>(u_pad_hat, u_pad);

  const auto test = [&](int dim) {
    const auto init2d = std::make_shared<azeban::TaylorVortex>();
    const auto initializer
        = std::make_shared<azeban::Init3DFrom2D>(dim, init2d);
    std::vector<zisa::int_t> Ns;
    std::vector<azeban::real_t> errs;
    for (zisa::int_t N = 16; N <= 128; N <<= 1) {
      azeban::Grid<3> grid(N);
      azeban::SmoothCutoff1D visc(0, 1);
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

      auto h_u_hat = zisa::array<azeban::complex_t, 4>(grid.shape_fourier(2));

      simulation.simulate_until(0.01);
      zisa::copy(h_u_hat, simulation.u());
      azeban::copy_to_padded(component(u_pad_hat, 0), component(h_u_hat, 0), 0);
      azeban::copy_to_padded(component(u_pad_hat, 1), component(h_u_hat, 1), 0);
      azeban::copy_to_padded(component(u_pad_hat, 2), component(h_u_hat, 2), 0);
      fft->backward();
      for (zisa::int_t i = 0; i < zisa::product(u_pad.shape()); ++i) {
        u_pad[i] /= zisa::product(u_pad.shape()) / u_pad.shape(0);
        u_pad[i] *= zisa::pow<3>(static_cast<azeban::real_t>(N_ref) / N);
      }

      azeban::real_t errL2 = 0;
      for (zisa::int_t i = 0; i < N_ref; ++i) {
	for (zisa::int_t j = 0; j < N_ref; ++j) {
	  for (zisa::int_t k = 0; k < N_ref; ++k) {
	    azeban::real_t u_ref_interp[3];
	    for (int d = 0 ; d < 3 ; ++d) {
	      const zisa::int_t d2d = d < dim ? d : d - 1;
	      const zisa::int_t i2d = dim > 0 ? i : j;
	      const zisa::int_t j2d = dim > 1 ? j : k;
	      if (d == dim) {
		u_ref_interp[d] = 0;
	      } else {
		u_ref_interp[d] = u_ref(d2d, i2d, j2d);
	      }
	    }
	    const azeban::real_t du = u_pad(0, i, j, k) - u_ref_interp[0];
	    const azeban::real_t dv = u_pad(1, i, j, k) - u_ref_interp[1];
	    const azeban::real_t dw = u_pad(2, i, j, k) - u_ref_interp[2];
	    const azeban::real_t err_loc = zisa::pow<2>(du) + zisa::pow<2>(dv) + zisa::pow<2>(dw);
	    errL2 += err_loc;
	  }
	}
      }
      errL2 = zisa::sqrt(errL2) / (N_ref * N_ref * N_ref);
      Ns.push_back(N);
      errs.push_back(errL2);
    }

    std::cout << "Taylor Vortex 3D L2 errors = [" << errs[0];
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
  const auto initializer = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const azeban::real_t conv_rate
      = measureConvergence<2>(initializer, 512, 0.25);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Double Shear Layer 3D const. x", "[slow]") {
  const auto initializer2d = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(0, initializer2d);
  const azeban::real_t conv_rate
      = measureConvergence<3>(initializer, 128, 0.25);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Double Shear Layer 3D const. y", "[slow]") {
  const auto initializer2d = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(1, initializer2d);
  const azeban::real_t conv_rate
      = measureConvergence<3>(initializer, 128, 0.25);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Double Shear Layer 3D const. z", "[slow]") {
  const auto initializer2d = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const auto initializer
      = std::make_shared<azeban::Init3DFrom2D>(2, initializer2d);
  const azeban::real_t conv_rate
      = measureConvergence<3>(initializer, 128, 0.25);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Taylor Green 2D", "[slow]") {
  const auto initializer = std::make_shared<azeban::TaylorGreen<2>>();
  const azeban::real_t conv_rate = measureConvergence<2>(initializer, 512, 1);
  REQUIRE(conv_rate >= 1);
}

TEST_CASE("Taylor Green 3D", "[slow]") {
  const auto initializer = std::make_shared<azeban::TaylorGreen<3>>();
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 128, 0.1);
  REQUIRE(conv_rate >= 1);
}
