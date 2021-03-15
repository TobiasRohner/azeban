#include <azeban/fft.hpp>
#include <azeban/init/initializer_factory.hpp>
#include <azeban/simulation_factory.hpp>
#include <cstdlib>
#include <fmt/core.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <zisa/config.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/io/hdf5_serial_writer.hpp>
#include <zisa/memory/array.hpp>

using namespace azeban;

template <int dim_v>
static void runFromConfig(const nlohmann::json &config) {
  if (!config.contains("time")) {
    fmt::print(stderr, "Config file does not contain \"time\"\n");
    exit(1);
  }
  const real_t t_final = config["time"];

  std::vector<real_t> snapshots;
  if (config.contains("snapshots")) {
    snapshots = config["snapshots"].get<std::vector<real_t>>();
  }
  if (!(snapshots.size() > 0 && snapshots.back() == t_final)) {
    snapshots.push_back(t_final);
  }

  std::string output = "result.h5";
  if (config.contains("output")) {
    output = config["output"];
  }

  auto simulation = make_simulation<complex_t, dim_v>(config);
  const auto &grid = simulation.grid();

  zisa::HDF5SerialWriter hdf5_writer(output);

  auto u_host
      = grid.make_array_phys(simulation.n_vars(), zisa::device_type::cpu);
  auto u_device
      = grid.make_array_phys(simulation.n_vars(), simulation.memory_location());
  auto u_hat_device = grid.make_array_fourier(simulation.n_vars(),
                                              simulation.memory_location());
  auto fft = make_fft<dim_v>(u_hat_device, u_device);

  auto initializer = make_initializer<dim_v>(config);
  initializer->initialize(simulation.u());

  zisa::copy(u_hat_device, simulation.u());
  fft->backward();
  zisa::copy(u_host, u_device);
  for (zisa::int_t i = 0; i < zisa::product(u_host.shape()); ++i) {
    u_host[i] /= zisa::product(u_host.shape()) / u_host.shape(0);
  }
  zisa::save(hdf5_writer, u_host, std::to_string(real_t(0)));
  for (real_t t : snapshots) {
    simulation.simulate_until(t);
    fmt::print("Time: {}\n", t);

    zisa::copy(u_hat_device, simulation.u());
    fft->backward();
    zisa::copy(u_host, u_device);
    for (zisa::int_t i = 0; i < zisa::product(u_host.shape()); ++i) {
      u_host[i] /= zisa::product(u_host.shape()) / u_host.shape(0);
    }
    zisa::save(hdf5_writer, u_host, std::to_string(t));
  }
}

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    fmt::print(stderr, "Usage: {} <config>\n", argv[0]);
    exit(1);
  }

  std::ifstream config_file(argv[1]);
  nlohmann::json config;
  config_file >> config;

  if (!config.contains("dimension")) {
    fmt::print(stderr, "Must provide dimension of simulation\n");
    exit(1);
  }
  int dim = config["dimension"];

  switch (dim) {
  case 1:
    runFromConfig<1>(config);
    break;
  case 2:
    runFromConfig<2>(config);
    break;
  case 3:
    runFromConfig<3>(config);
    break;
  default:
    fmt::print(stderr, "Invalid Dimension: {}\n", dim);
    exit(1);
  }

  return EXIT_SUCCESS;
}
