#ifndef EQUATION_FACTORY_H_
#define EQUATION_FACTORY_H_

#include <azeban/equations/burgers_factory.hpp>
#include <azeban/equations/equation.hpp>
#include <azeban/equations/incompressible_euler_factory.hpp>
#include <azeban/equations/incompressible_euler_naive_factory.hpp>
#include <azeban/equations/spectral_viscosity_factory.hpp>
#include <azeban/grid.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <string>

namespace azeban {

template <int Dim>
std::shared_ptr<Equation<Dim>> make_equation(const nlohmann::json &config,
                                             const Grid<Dim> &grid,
                                             bool has_tracer,
                                             zisa::device_type device) {
  if (!config.contains("name")) {
    fmt::print(stderr, "Equation config must contain key \"name\"\n");
    exit(1);
  }
  if (!config.contains("visc")) {
    fmt::print(stderr, "Equation config must contain key \"visc\"\n");
    exit(1);
  }
  if (!config["visc"].contains("type")) {
    fmt::print(stderr,
               "Must specify the type of Spectral Viscosity in key \"type\"\n");
    exit(1);
  }

  const std::string equation_name = config["name"];
  const std::string visc_type = config["visc"]["type"];

  if (visc_type == "Smooth Cutoff") {
    SmoothCutoff1D visc = make_smooth_cutoff_1d(config["visc"], grid);

    if (equation_name == "Burgers") {
      return make_burgers(grid, visc, device);
    } else if (equation_name == "Euler") {
      return make_incompressible_euler(grid, visc, has_tracer, device);
    } else if (equation_name == "Euler Naive") {
      return make_incompressible_euler_naive(grid, visc, device);
    } else {
      fmt::print(stderr, "Unknown Equation");
      exit(1);
    }
  } else if (visc_type == "Step") {
    Step1D visc = make_step_1d(config["visc"], grid);

    if (equation_name == "Burgers") {
      return make_burgers(grid, visc, device);
    } else if (equation_name == "Euler") {
      return make_incompressible_euler(grid, visc, has_tracer, device);
    } else if (equation_name == "Euler Naive") {
      return make_incompressible_euler_naive(grid, visc, device);
    } else {
      fmt::print(stderr, "Unknown Equation");
      exit(1);
    }
  } else if (visc_type == "Quadratic") {
    Quadratic visc = make_quadratic(config["visc"], grid);

    if (equation_name == "Burgers") {
      return make_burgers(grid, visc, device);
    } else if (equation_name == "Euler") {
      return make_incompressible_euler(grid, visc, has_tracer, device);
    } else if (equation_name == "Euler Naive") {
      return make_incompressible_euler_naive(grid, visc, device);
    } else {
      fmt::print(stderr, "Unknown Equation");
      exit(1);
    }
  } else {
    fmt::print(stderr, "Unknown Spectral Viscosity type\n");
    exit(1);
  }
  // Make compiler happy
  return nullptr;
}

}

#endif
