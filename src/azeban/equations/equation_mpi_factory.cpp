#include <azeban/equations/equation_mpi_factory.hpp>
#include <azeban/equations/incompressible_euler_mpi_factory.hpp>
#include <azeban/equations/spectral_viscosity_factory.hpp>
#include <azeban/forcing/no_forcing.hpp>
#include <azeban/forcing/white_noise_factory.hpp>
#include <fmt/core.h>
#include <string>

namespace azeban {

template <int Dim>
std::shared_ptr<Equation<Dim>> make_equation_mpi(const nlohmann::json &config,
                                                 const Grid<Dim> &grid,
                                                 MPI_Comm comm,
                                                 bool has_tracer) {
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
  std::string forcing_type = "No Forcing";
  if (config.contains("forcing")) {
    if (!config["forcing"].contains("type")) {
      fmt::print(stderr, "Must specify the type of Forcing in key \"type\"\n");
      exit(1);
    }
    forcing_type = config["forcing"]["type"];
  }

  if (visc_type == "Smooth Cutoff") {
    SmoothCutoff1D visc = make_smooth_cutoff_1d(config["visc"], grid);
    if (equation_name == "Euler") {
      if (forcing_type == "No Forcing") {
        return make_incompressible_euler_mpi(
            grid, comm, visc, NoForcing{}, has_tracer);
      } else if (forcing_type == "White Noise") {
        auto forcing = make_white_noise<std::mt19937>(config["forcing"], grid);
        return make_incompressible_euler_mpi(
            grid, comm, visc, forcing, has_tracer);
      }
    } else {
      fmt::print(stderr, "Unknown Equation");
      exit(1);
    }
  } else if (visc_type == "Step") {
    Step1D visc = make_step_1d(config["visc"], grid);
    if (equation_name == "Euler") {
      if (forcing_type == "No Forcing") {
        return make_incompressible_euler_mpi(
            grid, comm, visc, NoForcing{}, has_tracer);
      } else if (forcing_type == "White Noise") {
        auto forcing = make_white_noise<std::mt19937>(config["forcing"], grid);
        return make_incompressible_euler_mpi(
            grid, comm, visc, forcing, has_tracer);
      }
    } else {
      fmt::print(stderr, "Unknown Equation");
      exit(1);
    }
  } else if (visc_type == "Quadratic") {
    Quadratic visc = make_quadratic(config["visc"], grid);
    if (equation_name == "Euler") {
      if (forcing_type == "No Forcing") {
        return make_incompressible_euler_mpi(
            grid, comm, visc, NoForcing{}, has_tracer);
      } else if (forcing_type == "White Noise") {
        auto forcing = make_white_noise<std::mt19937>(config["forcing"], grid);
        return make_incompressible_euler_mpi(
            grid, comm, visc, forcing, has_tracer);
      }
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

template std::shared_ptr<Equation<2>>
make_equation_mpi<2>(const nlohmann::json &config,
                     const Grid<2> &grid,
                     MPI_Comm comm,
                     bool has_tracer);
template std::shared_ptr<Equation<3>>
make_equation_mpi<3>(const nlohmann::json &config,
                     const Grid<3> &grid,
                     MPI_Comm comm,
                     bool has_tracer);

}
