#ifndef TIMESTEP_INTEGRATOR_FACTORY_H_
#define TIMESTEP_INTEGRATOR_FACTORY_H_

#include <azeban/equations/equation.hpp>
#include <azeban/evolution/forward_euler.hpp>
#include <azeban/evolution/ssp_rk2.hpp>
#include <azeban/evolution/ssp_rk3.hpp>
#include <azeban/evolution/time_integrator.hpp>
#include <fmt/core.h>
#include <string>
#include <zisa/config.hpp>

namespace azeban {

template <typename Scalar, int Dim>
std::shared_ptr<TimeIntegrator<Scalar, Dim>>
make_timestepper(const auto &config,
                 const Grid<Dim> &grid,
                 const std::shared_ptr<Equation<Scalar, Dim>> &equation,
                 zisa::device_type device) {
  if (!config.contains("type")) {
    fmt::print(stderr, "Missing timestepper type\n");
    exit(1);
  }

  const std::string type = config["type"];
  if (type == "Forward Euler") {
    return std::make_shared<ForwardEuler<Scalar, Dim>>(
        device, grid.shape_fourier(equation->n_vars()), equation);
  } else if (type == "SSP RK2") {
    return std::make_shared<SSP_RK2<Scalar, Dim>>(
        device, grid.shape_fourier(equation->n_vars()), equation);
  } else if (type == "SSP RK3") {
    return std::make_shared<SSP_RK3<Scalar, Dim>>(
        device, grid.shape_fourier(equation->n_vars()), equation);
  } else {
    fmt::print(stderr, "Unknown time integrator: \"{}\"\n", type);
    exit(1);
  }
}

}

#endif