#if AZEBAN_HAS_MPI

#include <azeban/equations/equation_mpi_factory.hpp>
#include <azeban/equations/incompressible_euler_mpi_factory.hpp>
#if ZISA_HAS_CUDA
#include <azeban/equations/incompressible_euler_mpi_naive_factory.hpp>
#endif
#include <azeban/equations/spectral_viscosity_factory.hpp>
#include <azeban/forcing/boussinesq.hpp>
#include <azeban/forcing/no_forcing.hpp>
#include <azeban/forcing/sinusoidal_factory.hpp>
#include <azeban/forcing/white_noise_factory.hpp>
#include <fmt/core.h>
#include <string>

namespace azeban {

template <int Dim>
static std::shared_ptr<Equation<Dim>>
make_equation_mpi(const nlohmann::json &config,
                  const Grid<Dim> &grid,
                  const Communicator *comm,
                  bool has_tracer,
                  const std::string &visc_type,
                  const std::string &forcing_type,
                  const std::string &equation_name,
                  zisa::device_type device,
                  size_t seed) {
  if (visc_type == "Smooth Cutoff") {
    SmoothCutoff1D visc = make_smooth_cutoff_1d(config["visc"], grid);
    return make_equation_mpi<Dim>(config,
                                  grid,
                                  comm,
                                  has_tracer,
                                  visc,
                                  forcing_type,
                                  equation_name,
                                  device,
                                  seed);
  } else if (visc_type == "Step") {
    Step1D visc = make_step_1d(config["visc"], grid);
    return make_equation_mpi<Dim>(config,
                                  grid,
                                  comm,
                                  has_tracer,
                                  visc,
                                  forcing_type,
                                  equation_name,
                                  device,
                                  seed);
  } else if (visc_type == "Quadratic") {
    Quadratic visc = make_quadratic(config["visc"], grid);
    return make_equation_mpi<Dim>(config,
                                  grid,
                                  comm,
                                  has_tracer,
                                  visc,
                                  forcing_type,
                                  equation_name,
                                  device,
                                  seed);
  } else {
    fmt::print(stderr, "Unknown viscosity type: {}\n", visc_type);
    exit(1);
  }
  return nullptr;
}

template <int Dim, typename SpectralViscosity>
static std::shared_ptr<Equation<Dim>>
make_equation_mpi(const nlohmann::json &config,
                  const Grid<Dim> &grid,
                  const Communicator *comm,
                  bool has_tracer,
                  const SpectralViscosity &visc,
                  const std::string &forcing_type,
                  const std::string &equation_name,
                  zisa::device_type device,
                  size_t seed) {
  if (forcing_type == "No Forcing") {
    NoForcing forcing;
    return make_equation_mpi(
        grid, comm, has_tracer, visc, forcing, equation_name, device);
  } else if (forcing_type == "Sinusoidal") {
    Sinusoidal forcing = make_sinusoidal(config["forcing"], grid);
    return make_equation_mpi(
        grid, comm, has_tracer, visc, forcing, equation_name, device);
  } else if (forcing_type == "White Noise") {
    if (device == zisa::device_type::cpu) {
      WhiteNoise forcing
          = make_white_noise<std::mt19937>(config["forcing"], grid, seed);
      return make_equation_mpi(
          grid, comm, has_tracer, visc, forcing, equation_name, device);
    }
#if ZISA_HAS_CUDA
    else if (device == zisa::device_type::cuda) {
      WhiteNoise forcing = make_white_noise<curandStateXORWOW_t>(
          config["forcing"], grid, seed);
      return make_equation_mpi(
          grid, comm, has_tracer, visc, forcing, equation_name, device);
    }
#endif
    else {
      LOG_ERR("Unsupported device");
    }
  } else if (forcing_type == "Boussinesq") {
    Boussinesq forcing;
    return make_equation_mpi(
        grid, comm, has_tracer, visc, forcing, equation_name, device);
  } else {
    fmt::print(stderr, "Unknown forcing type: {}\n", forcing_type);
    exit(1);
  }
}

template <int Dim, typename SpectralViscosity, typename Forcing>
static std::shared_ptr<Equation<Dim>>
make_equation_mpi(const Grid<Dim> &grid,
                  const Communicator *comm,
                  bool has_tracer,
                  const SpectralViscosity &visc,
                  const Forcing &forcing,
                  const std::string &equation_name,
                  zisa::device_type device) {
  if (equation_name == "Euler") {
    return make_incompressible_euler_mpi<Dim>(
        grid, comm, visc, forcing, has_tracer, device);
  }
#if ZISA_HAS_CUDA
  else if (equation_name == "Euler Naive") {
    return make_incompressible_euler_mpi_naive<Dim>(
        grid, comm, visc, forcing, has_tracer);
  }
#endif
  else {
    fmt::print(stderr, "Unknown equation name: \"{}\"\n", equation_name);
    exit(1);
  }
  return nullptr;
}

template <int Dim>
std::shared_ptr<Equation<Dim>> make_equation_mpi(const nlohmann::json &config,
                                                 const Grid<Dim> &grid,
                                                 const Communicator *comm,
                                                 bool has_tracer,
                                                 zisa::device_type device,
                                                 size_t seed) {
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

  return make_equation_mpi(config,
                           grid,
                           comm,
                           has_tracer,
                           visc_type,
                           forcing_type,
                           equation_name,
                           device,
                           seed);
  // Make compiler happy
  return nullptr;
}

template std::shared_ptr<Equation<2>>
make_equation_mpi<2>(const nlohmann::json &config,
                     const Grid<2> &grid,
                     const Communicator *comm,
                     bool has_tracer,
                     zisa::device_type device,
                     size_t seed);
template std::shared_ptr<Equation<3>>
make_equation_mpi<3>(const nlohmann::json &config,
                     const Grid<3> &grid,
                     const Communicator *comm,
                     bool has_tracer,
                     zisa::device_type device,
                     size_t seed);

}

#endif
