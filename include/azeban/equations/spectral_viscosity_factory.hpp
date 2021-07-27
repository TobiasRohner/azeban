#ifndef SPECTRAL_VISCOSITY_FACTORY_H_
#define SPECTRAL_VISCOSITY_FACTORY_H_

#include "spectral_viscosity.hpp"
#include <azeban/grid.hpp>
#include <azeban/logging.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
SmoothCutoff1D make_smooth_cutoff_1d(const nlohmann::json &config,
                                     const Grid<Dim> &grid) {
  AZEBAN_ERR_IF(
      !config.contains("eps"),
      "Smooth Cutoff Viscosity expects \"eps\" parameter in config\n");

  AZEBAN_ERR_IF(!config.contains("k0"),
                "Smooth Cutoff Viscosity expects \"k0\" parameter in config\n");

  const real_t eps = config["eps"];
  const real_t k0 = config["k0"];

  return SmoothCutoff1D(eps / grid.N_phys, k0);
}

template <int Dim>
Step1D make_step_1d(const nlohmann::json &config, const Grid<Dim> &grid) {
  AZEBAN_ERR_IF(!config.contains("eps"),
                "Step Viscosity expects \"eps\" parameter in config\n");
  AZEBAN_ERR_IF(!config.contains("k0"),
                "Step Viscosity expects \"k0\" parameter in config\n");

  const real_t eps = config["eps"];
  const real_t k0 = config["k0"];
  return Step1D(eps / grid.N_phys, k0);
}

template <int Dim>
Quadratic make_quadratic(const nlohmann::json &config, const Grid<Dim> &grid) {
  AZEBAN_ERR_IF(!config.contains("eps"),
                "Quadratic Viscosity expects \"eps\" parameter in config\n");

  const real_t eps = config["eps"];
  return Quadratic(eps, grid.N_phys);
}

template <int Dim>
NoViscosity make_no_viscosity(const nlohmann::json &config,
                              const Grid<Dim> & /* grid */) {

  AZEBAN_ERR_IF(config["type"] != "None",
                "Config file did not request \"type = None\".\n");

  return NoViscosity();
}

}

#endif
