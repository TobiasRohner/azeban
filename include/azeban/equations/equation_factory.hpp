#ifndef EQUATION_FACTORY_H_
#define EQUATION_FACTORY_H_

#include <azeban/equations/burgers.hpp>
#include <azeban/equations/equation.hpp>
#include <azeban/equations/incompressible_euler.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/grid.hpp>
#include <fmt/core.h>
#include <string>

namespace azeban {

template <int Dim, typename Json>
std::shared_ptr<Equation<Dim>>
make_equation(Json &&config, const Grid<Dim> &grid, zisa::device_type device) {
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
    if (!config["visc"].contains("eps")) {
      fmt::print(
          stderr,
          "Smooth Cutoff Viscosity expects \"eps\" parameter in config\n");
      exit(1);
    }
    if (!config["visc"].contains("k0")) {
      fmt::print(
          stderr,
          "Smooth Cutoff Viscosity expects \"k0\" parameter in config\n");
      exit(1);
    }

    const real_t eps = config["visc"]["eps"];
    const real_t k0 = config["visc"]["k0"];
    SmoothCutoff1D visc(eps / grid.N_phys, k0);

    if (equation_name == "Burgers") {
      if constexpr (Dim == 1) {
        return std::make_shared<Burgers<SmoothCutoff1D>>(grid, visc, device);
      } else {
        fmt::print(stderr, "Burgers is only implemented for 1D\n");
        exit(1);
      }
    } else if (equation_name == "Euler") {
      if constexpr (Dim == 2 || Dim == 3) {
        return std::make_shared<IncompressibleEuler<Dim, SmoothCutoff1D>>(
            grid, visc, device);
      } else {
        fmt::print(stderr, "Euler is only implemented for 2D or 3D\n");
        exit(1);
      }
    }
  } else if (visc_type == "Step") {
    if (!config["visc"].contains("eps")) {
      fmt::print(stderr,
                 "Step Viscosity expects \"eps\" parameter in config\n");
      exit(1);
    }
    if (!config["visc"].contains("k0")) {
      fmt::print(stderr, "Step Viscosity expects \"k0\" parameter in config\n");
      exit(1);
    }
    const real_t eps = config["visc"]["eps"];
    const real_t k0 = config["visc"]["k0"];
    Step1D visc(eps / grid.N_phys, k0);

    if (equation_name == "Burgers") {
      if constexpr (Dim == 1) {
        return std::make_shared<Burgers<Step1D>>(grid, visc, device);
      } else {
        fmt::print(stderr, "Burgers is only implemented for 1D\n");
        exit(1);
      }
    } else if (equation_name == "Euler") {
      if constexpr (Dim == 2 || Dim == 3) {
        return std::make_shared<IncompressibleEuler<Dim, Step1D>>(
            grid, visc, device);
      } else {
        fmt::print(stderr, "Euler is only implemented for 2D or 3D\n");
        exit(1);
      }
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
