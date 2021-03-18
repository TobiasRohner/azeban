#ifndef SIMULATION_FACTORY_H_
#define SIMULATION_FACTORY_H_

#include <azeban/equations/equation_factory.hpp>
#include <azeban/evolution/cfl.hpp>
#include <azeban/evolution/time_integrator_factory.hpp>
#include <azeban/grid_factory.hpp>
#include <azeban/simulation.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <string>
#include <zisa/config.hpp>

namespace azeban {

template <int Dim>
Simulation<Dim> make_simulation(const nlohmann::json &config) {
  zisa::device_type device;
  if (!config.contains("device")) {
    fmt::print(stderr, "Device not specified. Defaulting to CPU");
    device = zisa::device_type::cpu;
  } else {
    const std::string device_name = config["device"];
    if (device_name == "cpu") {
      device = zisa::device_type::cpu;
    } else if (device_name == "cuda") {
      device = zisa::device_type::cuda;
    } else {
      fmt::print(stderr, "Unknown device type: {}\n", device_name);
      exit(1);
    }
  }

  if (!config.contains("grid")) {
    fmt::print(stderr, "Config must contain key \"grid\"\n");
    exit(1);
  }
  auto grid = make_grid<Dim>(config["grid"], device);

  if (!config.contains("equation")) {
    fmt::print(stderr, "Config must contain key \"equation\"\n");
    exit(1);
  }
  auto equation = make_equation<Dim>(config["equation"], grid, device);

  if (!config.contains("timestepper")) {
    fmt::print("Config is missing timestepper specifications\n");
    exit(1);
  }
  auto timestepper
      = make_timestepper(config["timestepper"], grid, equation, device);

  if (!config["timestepper"].contains("C")) {
    fmt::print(stderr, "Timestepper config is missing CFL constant \"C\"\n");
    exit(1);
  }
  const real_t C = config["timestepper"]["C"];
  auto cfl = CFL<Dim>(grid, C);

  return Simulation(
      grid.shape_fourier(equation->n_vars()), cfl, timestepper, device);
}

}

#endif
