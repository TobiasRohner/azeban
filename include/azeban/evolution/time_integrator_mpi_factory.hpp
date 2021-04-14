#ifndef TIMESTEP_INTEGRATOR_MPI_FACTORY_H_
#define TIMESTEP_INTEGRATOR_MPI_FACTORY_H_

#include <azeban/equations/equation.hpp>
#include <azeban/evolution/forward_euler.hpp>
#include <azeban/evolution/ssp_rk2.hpp>
#include <azeban/evolution/ssp_rk3.hpp>
#include <azeban/evolution/time_integrator.hpp>
#include <fmt/core.h>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <string>
#include <zisa/config.hpp>

namespace azeban {

template <int Dim, template <int> typename GridT>
std::shared_ptr<TimeIntegrator<Dim>>
make_timestepper_mpi(const nlohmann::json &config,
                     const GridT<Dim> &grid,
                     const std::shared_ptr<Equation<Dim>> &equation,
                     zisa::device_type device,
                     MPI_Comm comm) {
  if (!config.contains("type")) {
    fmt::print(stderr, "Missing timestepper type\n");
    exit(1);
  }

  const std::string type = config["type"];
  if (type == "Forward Euler") {
    return std::make_shared<ForwardEuler<Dim>>(
        device, grid.shape_fourier(equation->n_vars(), comm), equation);
  } else if (type == "SSP RK2") {
    return std::make_shared<SSP_RK2<Dim>>(
        device, grid.shape_fourier(equation->n_vars(), comm), equation);
  } else if (type == "SSP RK3") {
    return std::make_shared<SSP_RK3<Dim>>(
        device, grid.shape_fourier(equation->n_vars(), comm), equation);
  } else {
    fmt::print(stderr, "Unknown time integrator: \"{}\"\n", type);
    exit(1);
  }
}

}

#endif
