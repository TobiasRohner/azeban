#ifndef SIMULATION_MPI_FACTORY_H_
#define SIMULATION_MPI_FACTORY_H_

#include "simulation_factory.hpp"
#include <azeban/equations/equation_mpi_factory.hpp>
#include <azeban/grid_factory.hpp>

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
    if (!config.contains("grid")) {
      fmt::print(stderr, "Config must contain key \"grid\"\n");
      exit(1);
    }
    auto grid = make_grid<Dim>(config["grid"], zisa::device_type::cuda);

    if (!config.contains("equation")) {
      fmt::print(stderr, "Config must contain key \"equation\"\n");
      exit(1);
    }
    const bool has_tracer
        = config.contains("init") && config["init"].contains("tracer");
    auto equation
        = make_equation_mpi<Dim>(config["equation"], grid, comm, has_tracer);

    if (!config.contains("timestepper")) {
      fmt::print("Config is missing timestepper specifications\n");
      exit(1);
    }
    auto timestepper = make_timestepper(
        config["timestepper"], grid, equation, zisa::device_type::cpu);

    if (!config["timestepper"].contains("C")) {
      fmt::print(stderr, "Timestepper config is missing CFL constant \"C\"\n");
      exit(1);
    }
    const real_t C = config["timestepper"]["C"];
    auto cfl = CFL<Dim>(grid, C);

    return Simulation(grid.shape_fourier(equation->n_vars()),
                      cfl,
                      timestepper,
                      zisa::device_type::cpu);
  }
}

}

#endif
