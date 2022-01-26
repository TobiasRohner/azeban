#include <azeban/equations/burgers_factory.hpp>
#include <azeban/equations/equation_factory.hpp>
#include <azeban/equations/incompressible_euler_factory.hpp>
#include <azeban/equations/incompressible_euler_naive_factory.hpp>
#include <azeban/equations/spectral_viscosity_factory.hpp>
#include <azeban/forcing/no_forcing.hpp>
#include <azeban/forcing/white_noise_factory.hpp>
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
  std::string forcing_type = "No Forcing";
  if (config.contains("forcing")) {
    if (!config["forcing"].contains("type")) {
      fmt::print(stderr, "Must specify the type of Forcing in key \"type\"\n");
      exit(1);
    }
    forcing_type = config["forcing"]["type"];
  }

  auto make_equation = [&equation_name, &grid, &has_tracer, &device](
                           auto &&visc, auto &&forcing) {
    if (equation_name == "Burgers") {
      return make_burgers(grid, visc, device);
    } else if (equation_name == "Euler") {
      return make_incompressible_euler(grid, visc, forcing, has_tracer, device);
    } else if (equation_name == "Euler Naive") {
      return make_incompressible_euler_naive(grid, visc, device);
    }
    AZEBAN_ERR("Unkown Equation");
  };

  auto make_forcing_no_forcing = [&]() { return NoForcing{}; };
  auto make_forcing_white_noise_cpu = [&]() {
    return make_white_noise<std::mt19937>(config["forcing"], grid);
  };
#if ZISA_HAS_CUDA
  auto make_forcing_white_noise_cuda = [&]() {
    return make_white_noise<curandStateXORWOW_t>(config["forcing"], grid);
  };
#endif

#define REGISTER_EQUATION(                                                     \
    VISC_STR, VISC_FACTORY, FORCING_STR, FORCING_FACTORY)                      \
  if (visc_type == VISC_STR && forcing_type == FORCING_STR) {                  \
    auto visc = VISC_FACTORY(config["visc"], grid);                            \
    auto forcing = FORCING_FACTORY();                                          \
    return make_equation(visc, forcing);                                       \
  }

  REGISTER_EQUATION("Smooth Cutoff",
                    make_smooth_cutoff_1d,
                    "No Forcing",
                    make_forcing_no_forcing);
  REGISTER_EQUATION(
      "Step", make_step_1d, "No Forcing", make_forcing_no_forcing);
  REGISTER_EQUATION(
      "Quadratic", make_quadratic, "No Forcing", make_forcing_no_forcing);
  REGISTER_EQUATION(
      "None", make_no_viscosity, "No Forcing", make_forcing_no_forcing);
  if (device == zisa::device_type::cpu) {
    REGISTER_EQUATION("Smooth Cutoff",
                      make_smooth_cutoff_1d,
                      "White Noise",
                      make_forcing_white_noise_cpu);
    REGISTER_EQUATION(
        "Step", make_step_1d, "White Noise", make_forcing_white_noise_cpu);
    REGISTER_EQUATION("Quadratic",
                      make_quadratic,
                      "White Noise",
                      make_forcing_white_noise_cpu);
    REGISTER_EQUATION(
        "None", make_no_viscosity, "White Noise", make_forcing_white_noise_cpu);
  }
#if ZISA_HAS_CUDA
  else if (device == zisa::device_type::cuda) {
    REGISTER_EQUATION("Smooth Cutoff",
                      make_smooth_cutoff_1d,
                      "White Noise",
                      make_forcing_white_noise_cuda);
    REGISTER_EQUATION(
        "Step", make_step_1d, "White Noise", make_forcing_white_noise_cuda);
    REGISTER_EQUATION("Quadratic",
                      make_quadratic,
                      "White Noise",
                      make_forcing_white_noise_cuda);
    REGISTER_EQUATION("None",
                      make_no_viscosity,
                      "White Noise",
                      make_forcing_white_noise_cuda);
  }
#endif

#undef REGISTER_EQUATION

  AZEBAN_ERR("Invalid Equation Configuration\n");
}

template std::shared_ptr<Equation<1>>
make_equation<1>(const nlohmann::json &config,
                 const Grid<1> &grid,
                 bool has_tracer,
                 zisa::device_type device);
template std::shared_ptr<Equation<2>>
make_equation<2>(const nlohmann::json &config,
                 const Grid<2> &grid,
                 bool has_tracer,
                 zisa::device_type device);
template std::shared_ptr<Equation<3>>
make_equation<3>(const nlohmann::json &config,
                 const Grid<3> &grid,
                 bool has_tracer,
                 zisa::device_type device);

}
