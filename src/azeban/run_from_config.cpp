#include <azeban/init/initializer_factory.hpp>
#include <azeban/io/netcdf_snapshot_writer.hpp>
#include <azeban/io/writer_factory.hpp>
#include <azeban/run_from_config.hpp>
#include <azeban/sequence_factory.hpp>
#include <azeban/simulation.hpp>
#include <azeban/simulation_factory.hpp>
#include <cstdlib>
#include <fmt/core.h>
#include <limits>
#include <random>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/memory/array.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/mpi_types.hpp>
#include <azeban/operations/fft_mpi_factory.hpp>
#include <azeban/simulation_mpi_factory.hpp>
#include <mpi.h>
#endif

namespace azeban {

template <int dim_v>
static void run_from_config_impl(const nlohmann::json &config,
                                 zisa::int_t total_samples) {
  zisa::int_t num_samples = 1;
  if (config.contains("num_samples")) {
    num_samples = config["num_samples"];
  }

  zisa::int_t sample_idx_start = 0;
  if (config.contains("sample_idx_start")) {
    sample_idx_start = config["sample_idx_start"];
  }

  real_t time_offset = 0;
  if (config.contains("time_offset")) {
    time_offset = config["time_offset"];
  }

  std::mt19937 rng;
  size_t seed = 1;
  if (config.contains("seed")) {
    seed = config["seed"];
  }
  rng.seed(seed);

  auto simulation = make_simulation<dim_v>(config, seed);
  auto initializer = make_initializer<dim_v>(config, rng);

  if (!config.contains("writer")) {
    fmt::print(stderr, "Config file must contain key \"writer\"\n");
    exit(1);
  }
  std::string init_script;
  if (config.contains("init") && config["init"].contains("name")
      && config["init"]["name"] == "Python") {
    if (config["init"].contains("script")) {
      const std::string path = config["init"]["script"];
      std::ifstream ifs(path);
      std::ostringstream oss;
      oss << ifs.rdbuf();
      init_script = oss.str();
    }
  }
  auto writer = make_writer<dim_v>(config["writer"],
                                   simulation.grid(),
                                   simulation.n_vars() > dim_v,
                                   total_samples,
                                   sample_idx_start,
                                   config.dump(2, ' ', true),
                                   init_script);

  auto u_hat_out = simulation.grid().make_array_fourier(simulation.n_vars(),
                                                        zisa::device_type::cpu);
  auto u_out = simulation.grid().make_array_phys(simulation.n_vars(),
                                                 zisa::device_type::cpu);
  auto fft = make_fft<dim_v>(u_hat_out, u_out, FFT_BACKWARD);

  for (zisa::int_t sample = sample_idx_start;
       sample < sample_idx_start + num_samples;
       ++sample) {
    simulation.reset();
    simulation.set_time(time_offset);
    initializer->initialize(simulation.u());

    real_t t = 0;
    while ((t = writer->next_timestep())
           != std::numeric_limits<real_t>::infinity()) {
      simulation.simulate_until(t);
      fmt::print("Sample {}, Time {}\n", sample, t);
      zisa::copy(u_hat_out, simulation.u());
      writer->write(u_hat_out, t);
      scale(complex_t(1) / zisa::pow<dim_v>(simulation.grid().N_phys),
            u_hat_out.view());
      fft->backward();
      writer->write(u_out, t);
    }

    writer->reset();
  }
}

void run_from_config(const nlohmann::json &config, zisa::int_t total_samples) {
  if (!config.contains("dimension")) {
    fmt::print(stderr, "Must provide dimension of simulation\n");
    exit(1);
  }
  const int dim = config["dimension"];
  switch (dim) {
  case 1:
    run_from_config_impl<1>(config, total_samples);
    break;
  case 2:
    run_from_config_impl<2>(config, total_samples);
    break;
  case 3:
    run_from_config_impl<3>(config, total_samples);
    break;
  default:
    fmt::print(stderr, "Invalid Dimension: {}\n", dim);
    exit(1);
  }
}

#if AZEBAN_HAS_MPI
template <int dim_v>
static void run_from_config_MPI_impl(const nlohmann::json &config,
                                     zisa::int_t total_samples,
                                     const Communicator *comm) {
  const int rank = comm->rank();

  zisa::int_t num_samples = 1;
  if (config.contains("num_samples")) {
    num_samples = config["num_samples"];
  }

  zisa::int_t sample_idx_start = 0;
  if (config.contains("sample_idx_start")) {
    sample_idx_start = config["sample_idx_start"];
  }

  real_t time_offset = 0;
  if (config.contains("time_offset")) {
    time_offset = config["time_offset"];
  }

  std::mt19937 rng;
  size_t seed = 1 + rank;
  if (config.contains("seed")) {
    seed = config["seed"].get<size_t>() + rank;
  }
  rng.seed(seed);

  auto simulation = make_simulation_mpi<dim_v>(config, comm, seed);
  const auto &grid = simulation.grid();
  auto initializer = make_initializer<dim_v>(config, rng);

  if (!config.contains("writer")) {
    fmt::print(stderr, "Config file must contain key \"writer\"\n");
    exit(1);
  }
  std::string init_script;
  if (config.contains("init") && config["init"].contains("name")
      && config["init"]["name"] == "Python") {
    if (config["init"].contains("script")) {
      const std::string path = config["init"]["script"];
      std::ifstream ifs(path);
      std::ostringstream oss;
      oss << ifs.rdbuf();
      init_script = oss.str();
    }
  }
  auto writer = make_writer<dim_v>(config["writer"],
                                   simulation.grid(),
                                   simulation.n_vars() > dim_v,
                                   total_samples,
                                   sample_idx_start,
                                   config.dump(2, ' ', true),
                                   init_script,
                                   comm);

  for (zisa::int_t sample = sample_idx_start;
       sample < sample_idx_start + num_samples;
       ++sample) {
    simulation.reset();
    simulation.set_time(time_offset);
    initializer->initialize(
        simulation.u(), grid, comm, simulation.equation()->get_fft_work_area());

    auto u_hat_out_host = grid.make_array_fourier(
        simulation.n_vars(), zisa::device_type::cpu, comm);
    auto u_hat_out_device = grid.make_array_fourier(
        simulation.n_vars(), zisa::device_type::cuda, comm);
    auto u_out_host = grid.make_array_phys(
        simulation.n_vars(), zisa::device_type::cpu, comm);
    auto u_out_device = grid.make_array_phys(
        simulation.n_vars(), zisa::device_type::cuda, comm);
    auto fft = make_fft_mpi<dim_v>(
        u_hat_out_device, u_out_device, comm, FFT_BACKWARD);

    real_t t = 0;
    while ((t = writer->next_timestep())
           != std::numeric_limits<real_t>::infinity()) {
      simulation.simulate_until(t, comm);
      if (rank == 0) {
        fmt::print("Sample: {}, Time: {}\n", sample, t);
      }
      zisa::copy(u_hat_out_host, simulation.u());
      zisa::copy(u_hat_out_device, simulation.u());
      writer->write(u_hat_out_host, t, comm);
      scale(complex_t(1) / zisa::pow<dim_v>(grid.N_phys),
            u_hat_out_device.view());
      fft->backward();
      zisa::copy(u_out_host, u_out_device);
      writer->write(u_out_host, t, comm);
    }

    writer->reset();
  }
}

void run_from_config(const nlohmann::json &config,
                     zisa::int_t total_samples,
                     const Communicator *comm) {
  if (!config.contains("dimension")) {
    fmt::print(stderr, "Must provide dimension of simulation\n");
    exit(1);
  }
  const int dim = config["dimension"];
  switch (dim) {
  case 2:
    run_from_config_MPI_impl<2>(config, total_samples, comm);
    break;
  case 3:
    run_from_config_MPI_impl<3>(config, total_samples, comm);
    break;
  default:
    fmt::print(stderr, "Invalid Dimension: {}\n", dim);
    exit(1);
  }
}
#endif

}
