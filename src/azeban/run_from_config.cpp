#include <azeban/init/initializer_factory.hpp>
#include <azeban/io/netcdf_snapshot_writer.hpp>
#include <azeban/run_from_config.hpp>
#include <azeban/sequence_factory.hpp>
#include <azeban/simulation.hpp>
#include <azeban/simulation_factory.hpp>
#include <cstdlib>
#include <fmt/core.h>
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
static void run_from_config_impl(const nlohmann::json &config) {
  zisa::int_t num_samples = 1;
  if (config.contains("num_samples")) {
    num_samples = config["num_samples"];
  }

  zisa::int_t sample_idx_start = 0;
  if (config.contains("sample_idx_start")) {
    sample_idx_start = config["sample_idx_start"];
  }

  if (!config.contains("time")) {
    fmt::print(stderr, "Config file does not contain \"time\"\n");
    exit(1);
  }
  real_t t_final = config["time"];

  real_t time_offset = 0;
  if (config.contains("time_offset")) {
    time_offset = config["time_offset"];
  }

  std::vector<real_t> snapshots;
  if (config.contains("snapshots")) {
    snapshots = make_sequence<real_t>(config["snapshots"]);
  }
  if (snapshots.size() > 0 && snapshots.back() > t_final) {
    t_final = snapshots.back();
  }
  if (!(snapshots.size() > 0 && snapshots.back() == t_final)) {
    snapshots.push_back(t_final);
  }

  std::string output = "result";
  if (config.contains("output")) {
    output = config["output"];
  }

  std::mt19937 rng;
  size_t seed = 1;
  if (config.contains("seed")) {
    seed = config["seed"];
  }
  rng.seed(seed);

  auto simulation = make_simulation<dim_v>(config);
  auto initializer = make_initializer<dim_v>(config, rng);
  NetCDFSnapshotWriter<dim_v> writer(output);

  for (zisa::int_t sample = sample_idx_start;
       sample < sample_idx_start + num_samples;
       ++sample) {
    simulation.reset();
    simulation.set_time(time_offset);
    initializer->initialize(simulation.u());

    for (real_t t : snapshots) {
      simulation.simulate_until(t);
      fmt::print("Sample {}, Time {}\n", sample, t);
      writer.write_snapshot(simulation, sample);
    }
  }
}

void run_from_config(const nlohmann::json &config) {
  if (!config.contains("dimension")) {
    fmt::print(stderr, "Must provide dimension of simulation\n");
    exit(1);
  }
  const int dim = config["dimension"];
  switch (dim) {
  case 1:
    run_from_config_impl<1>(config);
    break;
  case 2:
    run_from_config_impl<2>(config);
    break;
  case 3:
    run_from_config_impl<3>(config);
    break;
  default:
    fmt::print(stderr, "Invalid Dimension: {}\n", dim);
    exit(1);
  }
}

#if AZEBAN_HAS_MPI
template <int dim_v>
static void run_from_config_MPI_impl(const nlohmann::json &config,
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

  if (!config.contains("time")) {
    fmt::print(stderr, "Config file does not contain \"time\"\n");
    exit(1);
  }
  real_t t_final = config["time"];

  real_t time_offset = 0;
  if (config.contains("time_offset")) {
    time_offset = config["time_offset"];
  }

  std::vector<real_t> snapshots;
  if (config.contains("snapshots")) {
    snapshots = make_sequence<real_t>(config["snapshots"]);
  }
  if (snapshots.size() > 0 && snapshots.back() > t_final) {
    t_final = snapshots.back();
  }
  if (!(snapshots.size() > 0 && snapshots.back() == t_final)) {
    snapshots.push_back(t_final);
  }

  std::string output = "result";
  if (config.contains("output")) {
    output = config["output"];
  }

  std::mt19937 rng;
  size_t seed = 1 * rank;
  if (config.contains("seed")) {
    seed = config["seed"].get<size_t>() * rank;
  }
  rng.seed(seed);

  auto simulation = make_simulation_mpi<dim_v>(config, comm);
  const auto &grid = simulation.grid();
  auto initializer = make_initializer<dim_v>(config, rng);
  NetCDFSnapshotWriter<dim_v> writer(output);

  for (zisa::int_t sample = sample_idx_start;
       sample < sample_idx_start + num_samples;
       ++sample) {
    simulation.reset();
    simulation.set_time(time_offset);
    initializer->initialize(
        simulation.u(), grid, comm, simulation.equation()->get_fft_work_area());

    for (real_t t : snapshots) {
      simulation.simulate_until(t, comm);
      if (rank == 0) {
        fmt::print("Sample: {}, Time: {}\n", sample, t);
      }
      writer.write_snapshot(simulation, sample, comm);
    }
  }
}

void run_from_config(const nlohmann::json &config, const Communicator *comm) {
  if (!config.contains("dimension")) {
    fmt::print(stderr, "Must provide dimension of simulation\n");
    exit(1);
  }
  const int dim = config["dimension"];
  switch (dim) {
  case 2:
    run_from_config_MPI_impl<2>(config, comm);
    break;
  case 3:
    run_from_config_MPI_impl<3>(config, comm);
    break;
  default:
    fmt::print(stderr, "Invalid Dimension: {}\n", dim);
    exit(1);
  }
}
#endif

}
