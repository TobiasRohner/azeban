#include <azeban/equations/equation_mpi_factory.hpp>
#include <azeban/equations/incompressible_euler_mpi_factory.hpp>
#include <azeban/equations/incompressible_euler_mpi_naive_factory.hpp>
#include <azeban/equations/spectral_viscosity_factory.hpp>
#include <azeban/forcing/no_forcing.hpp>
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
                  zisa::device_type device) {
  if (visc_type == "Smooth Cutoff") {
    SmoothCutoff1D visc = make_smooth_cutoff_1d(config["visc"], grid);
    return make_equation_mpi<Dim>(config,
                                  grid,
                                  comm,
                                  has_tracer,
                                  visc,
                                  forcing_type,
                                  equation_name,
                                  device);
  } else if (visc_type == "Step") {
    Step1D visc = make_step_1d(config["visc"], grid);
    return make_equation_mpi<Dim>(config,
                                  grid,
                                  comm,
                                  has_tracer,
                                  visc,
                                  forcing_type,
                                  equation_name,
                                  device);
  } else if (visc_type == "Quadratic") {
    Quadratic visc = make_quadratic(config["visc"], grid);
    return make_equation_mpi<Dim>(config,
                                  grid,
                                  comm,
                                  has_tracer,
                                  visc,
                                  forcing_type,
                                  equation_name,
                                  device);
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
                  zisa::device_type device) {
  if (forcing_type == "No Forcing") {
    NoForcing forcing;
    return make_equation_mpi(
        grid, comm, has_tracer, visc, forcing, equation_name, device);
  }
  if (forcing_type == "White Noise") {
    if (device == zisa::device_type::cpu) {
      WhiteNoise forcing
          = make_white_noise<std::mt19937>(config["forcing"], grid);
      return make_equation_mpi(
          grid, comm, has_tracer, visc, forcing, equation_name, device);
    }
#if ZISA_HAS_CUDA
    else if (device == zisa::device_type::cuda) {
      WhiteNoise forcing
          = make_white_noise<curandStateXORWOW_t>(config["forcing"], grid);
      return make_equation_mpi(
          grid, comm, has_tracer, visc, forcing, equation_name, device);
    }
#endif
    else {
      LOG_ERR("Unsupported device");
    }
  }
  return nullptr;
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
  } else if (equation_name == "Euler Naive") {
    return make_incompressible_euler_mpi_naive<Dim>(
        grid, comm, visc, forcing, has_tracer);
  } else {
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
                           device);
  // Make compiler happy
  return nullptr;
}

template std::shared_ptr<Equation<2>>
make_equation_mpi<2>(const nlohmann::json &config,
                     const Grid<2> &grid,
                     const Communicator *comm,
                     bool has_tracer,
                     zisa::device_type device);
template std::shared_ptr<Equation<3>>
make_equation_mpi<3>(const nlohmann::json &config,
                     const Grid<3> &grid,
                     const Communicator *comm,
                     bool has_tracer,
                     zisa::device_type device);

}
