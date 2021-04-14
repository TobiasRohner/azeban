#include <azeban/init/initializer_factory.hpp>
#include <azeban/operations/fft.hpp>
#include <azeban/profiler.hpp>
#include <azeban/simulation_factory.hpp>
#include <cstdlib>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fstream>
#include <hdf5.h>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <random>
#include <zisa/config.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/io/hdf5_serial_writer.hpp>
#include <zisa/memory/array.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi_types.hpp>
#include <azeban/simulation_mpi_factory.hpp>
#include <mpi.h>
#endif

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

  std::mt19937 rng;
  if (config.contains("seed")) {
    rng.seed(config["seed"].get<size_t>());
  }

  auto simulation = make_simulation<dim_v>(config);
  const auto &grid = simulation.grid();

  zisa::HDF5SerialWriter hdf5_writer(output);

  auto u_host
      = grid.make_array_phys(simulation.n_vars(), zisa::device_type::cpu);
  auto u_hat_host
      = grid.make_array_fourier(simulation.n_vars(), zisa::device_type::cpu);
  auto fft = make_fft<dim_v>(u_hat_host, u_host);

  auto initializer = make_initializer<dim_v>(config, rng);
  initializer->initialize(simulation.u());

  zisa::copy(u_hat_host, simulation.u());
  fft->backward();
  for (zisa::int_t i = 0; i < zisa::product(u_host.shape()); ++i) {
    u_host[i] /= zisa::product(u_host.shape()) / u_host.shape(0);
  }
  zisa::save(hdf5_writer, u_host, std::to_string(real_t(0)));
  for (real_t t : snapshots) {
    simulation.simulate_until(t);
    fmt::print("Time: {}\n", t);

    zisa::copy(u_hat_host, simulation.u());
    fft->backward();
    for (zisa::int_t i = 0; i < zisa::product(u_host.shape()); ++i) {
      u_host[i] /= zisa::product(u_host.shape()) / u_host.shape(0);
    }
    zisa::save(hdf5_writer, u_host, std::to_string(t));
  }
}

#if AZEBAN_HAS_MPI
template <int dim_v>
static void runFromConfig_MPI(const nlohmann::json &config, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

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

  std::mt19937 rng;
  if (config.contains("seed")) {
    rng.seed(config["seed"].get<size_t>());
  }

  auto simulation = make_simulation_mpi<dim_v>(config, comm);
  const auto &grid = simulation.grid();

  std::unique_ptr<zisa::HDF5SerialWriter> hdf5_writer;
  if (rank == 0) {
    hdf5_writer = std::make_unique<zisa::HDF5SerialWriter>(output);
  }

  auto u_host
      = grid.make_array_phys(simulation.n_vars(), zisa::device_type::cpu, comm);
  auto u_device = grid.make_array_phys(
      simulation.n_vars(), zisa::device_type::cuda, comm);
  auto u_hat_device = grid.make_array_fourier(
      simulation.n_vars(), zisa::device_type::cuda, comm);
  auto fft = make_fft_mpi<dim_v>(u_hat_device, u_device, comm);

  auto initializer = make_initializer<dim_v>(config, rng);
  zisa::array<real_t, dim_v + 1> u_init;
  if (rank == 0) {
    u_init = grid.make_array_phys(simulation.n_vars(), zisa::device_type::cpu);
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

  zisa::copy(u_hat_device, simulation.u());
  fft->backward();
  zisa::copy(u_host, u_device);
  for (zisa::int_t i = 0; i < zisa::product(u_host.shape()); ++i) {
    u_host[i] /= zisa::pow<dim_v>(grid.N_phys);
  }
  for (zisa::int_t i = 0; i < simulation.n_vars(); ++i) {
    MPI_Igatherv(u_host.raw() + i * n_elems_per_component_loc,
                 cnts[rank],
                 mpi_type<real_t>(),
                 u_init.raw() + i * n_elems_per_component_glob,
                 cnts.data(),
                 displs.data(),
                 mpi_type<real_t>(),
                 0,
                 comm,
                 &reqs[i]);
  }
  MPI_Waitall(simulation.n_vars(), reqs.data(), MPI_STATUSES_IGNORE);
  if (rank == 0) {
    zisa::save(*hdf5_writer, u_init, std::to_string(real_t(0)));
  }
  for (real_t t : snapshots) {
    simulation.simulate_until(t, comm);
    if (rank == 0) {
      fmt::print("Time: {}\n", t);
    }

    zisa::copy(u_hat_device, simulation.u());
    fft->backward();
    zisa::copy(u_host, u_device);
    for (zisa::int_t i = 0; i < zisa::product(u_host.shape()); ++i) {
      u_host[i] /= zisa::pow<dim_v>(grid.N_phys);
    }
    for (zisa::int_t i = 0; i < simulation.n_vars(); ++i) {
      MPI_Igatherv(u_host.raw() + i * n_elems_per_component_loc,
                   cnts[rank],
                   mpi_type<real_t>(),
                   u_init.raw() + i * n_elems_per_component_glob,
                   cnts.data(),
                   displs.data(),
                   mpi_type<real_t>(),
                   0,
                   comm,
                   &reqs[i]);
    }
    MPI_Waitall(simulation.n_vars(), reqs.data(), MPI_STATUSES_IGNORE);
    if (rank == 0) {
      zisa::save(*hdf5_writer, u_init, std::to_string(t));
    }
  }
}
#endif

int main(int argc, char *argv[]) {
#if AZEBAN_HAS_MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  LOG_ERR_IF(provided < MPI_THREAD_FUNNELED,
             "MPI did not provide enough thread safety");
#endif

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

#if AZEBAN_HAS_MPI
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const bool use_mpi = size > 1;
#endif

  Profiler::start();
#if AZEBAN_HAS_MPI
  if (use_mpi) {
    switch (dim) {
    case 2:
      runFromConfig_MPI<2>(config, MPI_COMM_WORLD);
      break;
    case 3:
      runFromConfig_MPI<3>(config, MPI_COMM_WORLD);
      break;
    default:
      fmt::print(stderr, "Invalid dimension: {}\n", dim);
      exit(-1);
    }
  } else {
#endif
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
#if AZEBAN_HAS_MPI
  }
#endif
  Profiler::stop();

#if AZEBAN_DO_PROFILE
  fmt::print(Profiler::summary());
  std::ofstream pstream("profiling.json");
  pstream << std::setw(2) << Profiler::json();
#endif

#if AZEBAN_HAS_MPI
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}
