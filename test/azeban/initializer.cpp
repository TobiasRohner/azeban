#include <azeban/catch.hpp>

#include <vector>
#include <iostream>
#include <azeban/random/delta.hpp>
#include <azeban/init/brownian_motion.hpp>
#include <azeban/init/discontinuous_vortex_patch.hpp>
#include <azeban/init/double_shear_layer.hpp>
#include <azeban/init/init_3d_from_2d.hpp>
#include <azeban/init/shear_tube.hpp>
#include <azeban/init/taylor_green.hpp>
#include <azeban/init/taylor_vortex.hpp>

template<int dim_v>
static azeban::real_t measureConvergence(const std::shared_ptr<azeban::Initializer<dim_v>> &initializer, zisa::int_t N_ref) {
  zisa::shape_t<dim_v + 1> shape_ref;
  shape_ref[0] = dim_v;
  for (zisa::int_t i = 1 ; i <= dim_v ; ++i) {
    shape_ref[i] = N_ref;
  }
  auto u_ref = zisa::array<azeban::real_t, dim_v + 1>(shape_ref);
  initializer->initialize(u_ref);

  std::vector<zisa::int_t> Ns;
  std::vector<azeban::real_t> errs;
  for (zisa::int_t N = 16 ; N < N_ref ; N <<= 1) {
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = dim_v;
    for (zisa::int_t i = 1 ; i <= dim_v ; ++i) {
      shape[i] = N;
    }
    auto u = zisa::array<azeban::real_t, dim_v + 1>(shape);
    initializer->initialize(u);
    azeban::real_t errL2 = 0;
    if constexpr (dim_v == 2) {
      for (zisa::int_t i = 0; i < N_ref; ++i) {
	for (zisa::int_t j = 0; j < N_ref; ++j) {
	  const zisa::int_t i_sol = i * N / N_ref;
	  const zisa::int_t j_sol = j * N / N_ref;
	  const azeban::real_t u_ref_interp = u_ref(0, i, j);
	  const azeban::real_t v_ref_interp = u_ref(1, i, j);
	  const azeban::real_t du = u(0, i_sol, j_sol) - u_ref_interp;
	  const azeban::real_t dv = u(1, i_sol, j_sol) - v_ref_interp;
	  const azeban::real_t err_loc = zisa::pow<2>(du) + zisa::pow<2>(dv);
	  errL2 += err_loc;
	}
      }
      errL2 = zisa::sqrt(errL2) / (N_ref * N_ref);
    } else {
      for (zisa::int_t i = 0; i < N_ref; ++i) {
	for (zisa::int_t j = 0; j < N_ref; ++j) {
	  for (zisa::int_t k = 0; k < N_ref; ++k) {
	    const zisa::int_t i_sol = i * N / N_ref;
	    const zisa::int_t j_sol = j * N / N_ref;
	    const zisa::int_t k_sol = k * N / N_ref;
	    const azeban::real_t u_ref_interp = u_ref(0, i, j, k);
	    const azeban::real_t v_ref_interp = u_ref(1, i, j, k);
	    const azeban::real_t w_ref_interp = u_ref(1, i, j, k);
	    const azeban::real_t du = u(0, i_sol, j_sol, k_sol) - u_ref_interp;
	    const azeban::real_t dv = u(1, i_sol, j_sol, k_sol) - v_ref_interp;
	    const azeban::real_t dw = u(2, i_sol, j_sol, k_sol) - w_ref_interp;
	    const azeban::real_t err_loc = zisa::pow<2>(du) + zisa::pow<2>(dv) + zisa::pow<2>(dw);
	    errL2 += err_loc;
	  }
	}
      }
      errL2 = zisa::sqrt(errL2) / (N_ref * N_ref * N_ref);
    }
    Ns.push_back(N);
    errs.push_back(errL2);
  }

  std::cout << "L2 errors = [" << errs[0];
  for (zisa::int_t i = 1 ; i < errs.size() ; ++i) {
    std::cout << ", " << errs[i];
  }
  std::cout << "]" << std::endl;

  const azeban::real_t conv_rate = (zisa::log(errs[0]) - zisa::log(errs[errs.size() - 1])) / zisa::log(Ns[Ns.size() - 1] / Ns[0]);
  std::cout << "Estimated convergence rate: " << conv_rate << std::endl;

  return conv_rate;
}



TEST_CASE("Convergence Discontinuous Vortex Patch 2D", "[slow]") {
  const auto initializer = std::make_shared<azeban::DiscontinuousVortexPatch>();
  const azeban::real_t conv_rate = measureConvergence<2>(initializer, 1024);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Discontinuous Vortex Patch 3D const. x", "[slow]") {
  const auto initializer2d = std::make_shared<azeban::DiscontinuousVortexPatch>();
  const auto initializer = std::make_shared<azeban::Init3DFrom2D>(0, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Discontinuous Vortex Patch 3D const. y", "[slow]") {
  const auto initializer2d = std::make_shared<azeban::DiscontinuousVortexPatch>();
  const auto initializer = std::make_shared<azeban::Init3DFrom2D>(1, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Discontinuous Vortex Patch 3D const. z", "[slow]") {
  const auto initializer2d = std::make_shared<azeban::DiscontinuousVortexPatch>();
  const auto initializer = std::make_shared<azeban::Init3DFrom2D>(2, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Double Shear Layer 2D", "[slow]") {
  const auto initializer = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const azeban::real_t conv_rate = measureConvergence<2>(initializer, 1024);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Double Shear Layer 3D const. x", "[slow]") {
  const auto initializer2d = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const auto initializer = std::make_shared<azeban::Init3DFrom2D>(0, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Double Shear Layer 3D const. y", "[slow]") {
  const auto initializer2d = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const auto initializer = std::make_shared<azeban::Init3DFrom2D>(1, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Double Shear Layer 3D const. z", "[slow]") {
  const auto initializer2d = std::make_shared<azeban::DoubleShearLayer>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const auto initializer = std::make_shared<azeban::Init3DFrom2D>(2, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Shear Tube 2D", "[slow]") {
  const auto initializer = std::make_shared<azeban::ShearTube>(
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.2)),
      azeban::RandomVariable<azeban::real_t>(
          std::make_shared<azeban::Delta<azeban::real_t>>(0.05)));
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Taylor Green 2D", "[slow]") {
  const auto initializer = std::make_shared<azeban::TaylorGreen<2>>();
  const azeban::real_t conv_rate = measureConvergence<2>(initializer, 1024);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Taylor Green 3D", "[slow]") {
  const auto initializer = std::make_shared<azeban::TaylorGreen<3>>();
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Taylor Vortex 2D", "[slow]") {
  const auto initializer = std::make_shared<azeban::TaylorVortex>();
  const azeban::real_t conv_rate = measureConvergence<2>(initializer, 1024);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Taylor Vortex 3D const. x", "[slow]") {
  const auto initializer2d = std::make_shared<azeban::TaylorVortex>();
  const auto initializer = std::make_shared<azeban::Init3DFrom2D>(0, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Taylor Vortex 3D const. y", "[slow]") {
  const auto initializer2d = std::make_shared<azeban::TaylorVortex>();
  const auto initializer = std::make_shared<azeban::Init3DFrom2D>(1, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}

TEST_CASE("Convergence Taylor Vortex 3D const. z", "[slow]") {
  const auto initializer2d = std::make_shared<azeban::TaylorVortex>();
  const auto initializer = std::make_shared<azeban::Init3DFrom2D>(2, initializer2d);
  const azeban::real_t conv_rate = measureConvergence<3>(initializer, 512);
  REQUIRE(conv_rate > 1);
}
