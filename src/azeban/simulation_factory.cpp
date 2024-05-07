#include <azeban/equations/equation_factory.hpp>
#include <azeban/evolution/time_integrator_factory.hpp>
#include <azeban/grid_factory.hpp>
#include <azeban/simulation_factory.hpp>

namespace azeban {

template <int Dim>
Simulation<Dim> make_simulation(const nlohmann::json &config, size_t seed) {
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
  const bool has_tracer
      = config.contains("init") && config["init"].contains("tracer");
  auto equation
      = make_equation<Dim>(config["equation"], grid, has_tracer, device, seed);

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

  return Simulation<Dim>(grid, C, timestepper, device);
}

template Simulation<1> make_simulation<1>(const nlohmann::json &config,
                                          size_t seed);
template Simulation<2> make_simulation<2>(const nlohmann::json &config,
                                          size_t seed);
template Simulation<3> make_simulation<3>(const nlohmann::json &config,
                                          size_t seed);

}
