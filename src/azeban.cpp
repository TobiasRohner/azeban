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

int main(int argc, const char *argv[]) {
  static constexpr int dim_v = 2;
  static constexpr azeban::real_t t_final = 1;
  static constexpr int n_frames = 60;

  if (argc != 2) {
    fmt::print(stderr, "Usage: {} <config>\n", argv[0]);
    exit(1);
  }

  std::ifstream config_file(argv[1]);
  nlohmann::json config;
  config_file >> config;

  auto simulation = make_simulation<azeban::complex_t, dim_v>(config);
  auto initializer = make_initializer<dim_v>(config);

  initializer->initialize(simulation.u());

  const auto &grid = simulation.grid();

  zisa::HDF5SerialWriter hdf5_writer("result.hdf5");

  auto u_host
      = grid.make_array_phys(simulation.n_vars(), zisa::device_type::cpu);
  auto u_device
      = grid.make_array_phys(simulation.n_vars(), zisa::device_type::cuda);

  auto fft = make_fft<dim_v>(simulation.u(), u_device);

  fft->backward();
  zisa::copy(u_host, u_device);
  zisa::save(hdf5_writer, u_host, std::to_string(0));
  for (int i = 0; i < n_frames; ++i) {
    std::cerr << i << std::endl;
    simulation.simulate_for(t_final / n_frames);

    fft->backward();
    zisa::copy(u_host, u_device);
    for (zisa::int_t i = 0; i < zisa::product(u_host.shape()); ++i) {
      u_host[i] /= zisa::product(u_host.shape()) / u_host.shape(0);
    }
    zisa::save(hdf5_writer, u_host, std::to_string(i + 1));
  }

  return EXIT_SUCCESS;
}
