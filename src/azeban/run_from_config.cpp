#include <azeban/init/initializer_factory.hpp>
#include <azeban/io/netcdf_snapshot_writer.hpp>
#include <azeban/operations/fft_mpi_factory.hpp>
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
#include <azeban/mpi_types.hpp>
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
  if (config.contains("seed")) {
    rng.seed(config["seed"].get<size_t>());
  }

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
                                     MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

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
  if (config.contains("seed")) {
    rng.seed(config["seed"].get<size_t>());
  }

  auto simulation = make_simulation_mpi<dim_v>(config, comm);
  const auto &grid = simulation.grid();
  auto initializer = make_initializer<dim_v>(config, rng);
  NetCDFSnapshotWriter<dim_v> writer(output);

  auto u_host
      = grid.make_array_phys(simulation.n_vars(), zisa::device_type::cpu, comm);
  auto u_device = grid.make_array_phys(
      simulation.n_vars(), zisa::device_type::cuda, comm);
  auto u_hat_device = grid.make_array_fourier(
      simulation.n_vars(), zisa::device_type::cuda, comm);
  auto fft = make_fft_mpi<dim_v>(u_hat_device,
                                 u_device,
                                 comm,
                                 FFT_FORWARD,
                                 simulation.equation()->get_fft_work_area());

  for (zisa::int_t sample = sample_idx_start;
       sample < sample_idx_start + num_samples;
       ++sample) {
    simulation.reset();
    simulation.set_time(time_offset);
    zisa::array<real_t, dim_v + 1> u_init;
    if (rank == 0) {
      u_init
          = grid.make_array_phys(simulation.n_vars(), zisa::device_type::cpu);
      initializer->initialize(u_init);
    }
    std::vector<int> cnts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
      cnts[r] = zisa::pow<dim_v - 1>(grid.N_phys)
                * (grid.N_phys / size
                   + (zisa::integer_cast<zisa::int_t>(r) < grid.N_phys % size));
    }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) {
      displs[r] = displs[r - 1] + cnts[r - 1];
    }
    std::vector<MPI_Request> reqs(simulation.n_vars());
    const zisa::int_t n_elems_per_component_glob
        = zisa::product(grid.shape_phys(1));
    const zisa::int_t n_elems_per_component_loc
        = zisa::product(grid.shape_phys(1, comm));
    for (zisa::int_t i = 0; i < simulation.n_vars(); ++i) {
      MPI_Iscatterv(u_init.raw() + i * n_elems_per_component_glob,
                    cnts.data(),
                    displs.data(),
                    mpi_type<real_t>(),
                    u_host.raw() + i * n_elems_per_component_loc,
                    cnts[rank],
                    mpi_type<real_t>(),
                    0,
                    comm,
                    &reqs[i]);
    }
    MPI_Waitall(simulation.n_vars(), reqs.data(), MPI_STATUSES_IGNORE);
    zisa::copy(u_device, u_host);
    fft->forward();
    zisa::copy(simulation.u(), u_hat_device);

    for (real_t t : snapshots) {
      simulation.simulate_until(t, comm);
      if (rank == 0) {
        fmt::print("Sample: {}, Time: {}\n", sample, t);
      }
      writer.write_snapshot(simulation, sample, comm);
    }
  }
}

void run_from_config(const nlohmann::json &config, MPI_Comm comm) {
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
