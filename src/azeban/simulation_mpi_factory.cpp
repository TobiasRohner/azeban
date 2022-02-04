#include <azeban/equations/equation_mpi_factory.hpp>
#include <azeban/evolution/time_integrator_mpi_factory.hpp>
#include <azeban/grid_factory.hpp>
#include <azeban/simulation_factory.hpp>
#include <azeban/simulation_mpi_factory.hpp>

namespace azeban {

template <int Dim>
Simulation<Dim> make_simulation_mpi(const nlohmann::json &config,
                                    MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  if (size == 1) {
    // No MPI
    return make_simulation<Dim>(config);
  } else {
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
    auto equation = make_equation_mpi<Dim>(
        config["equation"], grid, comm, has_tracer, device);

    if (!config.contains("timestepper")) {
      fmt::print("Config is missing timestepper specifications\n");
      exit(1);
    }
    auto timestepper = make_timestepper_mpi(
        config["timestepper"], grid, equation, device, comm);

    if (!config["timestepper"].contains("C")) {
      fmt::print(stderr, "Timestepper config is missing CFL constant \"C\"\n");
      exit(1);
    }
    const real_t C = config["timestepper"]["C"];
    auto cfl = CFL<Dim>(grid, C);

    return Simulation(
        grid.shape_fourier(equation->n_vars(), comm), cfl, timestepper, device);
  }
}

template Simulation<2> make_simulation_mpi<2>(const nlohmann::json &config,
                                              MPI_Comm comm);
template Simulation<3> make_simulation_mpi<3>(const nlohmann::json &config,
                                              MPI_Comm comm);

}
