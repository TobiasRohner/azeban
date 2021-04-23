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
#include <zisa/io/netcdf_file.hpp>
#include <zisa/io/netcdf_serial_writer.hpp>
#include <zisa/memory/array.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi_types.hpp>
#include <azeban/simulation_mpi_factory.hpp>
#include <mpi.h>
#endif

using namespace azeban;

template <int Dim>
zisa::NetCDFSerialWriter make_nc_writer(const std::string &filename,
                                        zisa::int_t num_samples,
                                        const Simulation<Dim> &simulation,
                                        const std::vector<real_t> &snapshots);

template <>
zisa::NetCDFSerialWriter
make_nc_writer<1>(const std::string &filename,
                  zisa::int_t num_samples,
                  const Simulation<1> &simulation,
                  const std::vector<real_t> &snapshots) {
  const Grid<1> &grid = simulation.grid();

  using nc_dim_t = std::tuple<std::string, size_t>;
  using nc_var_t
      = std::tuple<std::string, std::vector<std::string>, zisa::ErasedDataType>;
  std::vector<nc_dim_t> dims;
  std::vector<nc_var_t> vars;

  dims.emplace_back("timesteps", snapshots.size());
  dims.emplace_back("N", grid.N_phys);

  vars.emplace_back("time",
                    std::vector<std::string>{"timesteps"},
                    zisa::erase_data_type<real_t>());
  for (zisa::int_t sample = 0; sample < num_samples; ++sample) {
    for (real_t t : snapshots) {
      std::string name
          = "sample_" + std::to_string(sample) + "_time_" + std::to_string(t);
      vars.emplace_back(name + "_u",
                        std::vector<std::string>{"N"},
                        zisa::erase_data_type<real_t>());
    }
  }

  zisa::NetCDFFileStructure file_structure(dims, vars);
  return zisa::NetCDFSerialWriter(filename, file_structure);
}

template <>
zisa::NetCDFSerialWriter
make_nc_writer<2>(const std::string &filename,
                  zisa::int_t num_samples,
                  const Simulation<2> &simulation,
                  const std::vector<real_t> &snapshots) {
  const Grid<2> &grid = simulation.grid();

  using nc_dim_t = std::tuple<std::string, size_t>;
  using nc_var_t
      = std::tuple<std::string, std::vector<std::string>, zisa::ErasedDataType>;
  std::vector<nc_dim_t> dims;
  std::vector<nc_var_t> vars;

  dims.emplace_back("timesteps", snapshots.size());
  dims.emplace_back("N", grid.N_phys);

  vars.emplace_back("time",
                    std::vector<std::string>{"timesteps"},
                    zisa::erase_data_type<real_t>());
  for (zisa::int_t sample = 0; sample < num_samples; ++sample) {
    for (real_t t : snapshots) {
      std::string name
          = "sample_" + std::to_string(sample) + "_time_" + std::to_string(t);
      vars.emplace_back(name + "_u",
                        std::vector<std::string>{"N", "N"},
                        zisa::erase_data_type<real_t>());
      vars.emplace_back(name + "_v",
                        std::vector<std::string>{"N", "N"},
                        zisa::erase_data_type<real_t>());
    }
  }

  zisa::NetCDFFileStructure file_structure(dims, vars);
  return zisa::NetCDFSerialWriter(filename, file_structure);
}

template <>
zisa::NetCDFSerialWriter
make_nc_writer<3>(const std::string &filename,
                  zisa::int_t num_samples,
                  const Simulation<3> &simulation,
                  const std::vector<real_t> &snapshots) {
  const Grid<3> &grid = simulation.grid();
  using nc_dim_t = std::tuple<std::string, size_t>;
  using nc_var_t
      = std::tuple<std::string, std::vector<std::string>, zisa::ErasedDataType>;
  std::vector<nc_dim_t> dims;
  std::vector<nc_var_t> vars;

  dims.emplace_back("timesteps", snapshots.size());
  dims.emplace_back("N", grid.N_phys);

  vars.emplace_back("time",
                    std::vector<std::string>{"timesteps"},
                    zisa::erase_data_type<real_t>());
  for (zisa::int_t sample = 0; sample < num_samples; ++sample) {
    for (real_t t : snapshots) {
      std::string name
          = "sample_" + std::to_string(sample) + "_time_" + std::to_string(t);
      vars.emplace_back(name + "_u",
                        std::vector<std::string>{"N", "N", "N"},
                        zisa::erase_data_type<real_t>());
      vars.emplace_back(name + "_v",
                        std::vector<std::string>{"N", "N", "N"},
                        zisa::erase_data_type<real_t>());
      vars.emplace_back(name + "_w",
                        std::vector<std::string>{"N", "N", "N"},
                        zisa::erase_data_type<real_t>());
    }
  }

  zisa::NetCDFFileStructure file_structure(dims, vars);
  return zisa::NetCDFSerialWriter(filename, file_structure);
}

template <int dim_v>
static void runFromConfig(const nlohmann::json &config) {
  zisa::int_t num_samples = 1;
  if (config.contains("num_samples")) {
    num_samples = config["num_samples"];
  }

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

  std::string output = "result";
  if (config.contains("output")) {
    output = config["output"];
  }

  std::mt19937 rng;
  if (config.contains("seed")) {
    rng.seed(config["seed"].get<size_t>());
  }

  auto simulation = make_simulation<dim_v>(config);
  const auto &grid = simulation.grid();
  auto initializer = make_initializer<dim_v>(config, rng);

  auto writer = make_nc_writer<dim_v>(
      output + ".nc", num_samples, simulation, snapshots);
  zisa::save(
      writer,
      zisa::array_const_view<real_t, 1>(zisa::shape_t<1>(snapshots.size()),
                                        snapshots.data(),
                                        zisa::device_type::cpu),
      "time");

  for (zisa::int_t sample = 0; sample < num_samples; ++sample) {
    auto u_host
        = grid.make_array_phys(simulation.n_vars(), zisa::device_type::cpu);
    auto u_hat_host
        = grid.make_array_fourier(simulation.n_vars(), zisa::device_type::cpu);
    auto fft = make_fft<dim_v>(u_hat_host, u_host);

    simulation.reset();
    initializer->initialize(simulation.u());

    for (real_t t : snapshots) {
      simulation.simulate_until(t);
      fmt::print("Sample {}, Time {}\n", sample, t);

      zisa::copy(u_hat_host, simulation.u());
      fft->backward();
      for (zisa::int_t i = 0; i < zisa::product(u_host.shape()); ++i) {
        u_host[i] /= zisa::product(u_host.shape()) / u_host.shape(0);
      }
      const std::string name
          = "sample_" + std::to_string(sample) + "_time_" + std::to_string(t);
      if (dim_v == 1) {
        zisa::shape_t<1> slice_shape(grid.N_phys);
        zisa::array_const_view<real_t, 1> u(
            slice_shape, u_host.raw(), zisa::device_type::cpu);
        zisa::save(writer, u, name + "_u");
      }
      if (dim_v == 2) {
        zisa::shape_t<2> slice_shape(grid.N_phys, grid.N_phys);
        zisa::array_const_view<real_t, 2> u(
            slice_shape, u_host.raw(), zisa::device_type::cpu);
        zisa::array_const_view<real_t, 2> v(slice_shape,
                                            u_host.raw()
                                                + zisa::product(slice_shape),
                                            zisa::device_type::cpu);
        zisa::save(writer, u, name + "_u");
        zisa::save(writer, v, name + "_v");
      }
      if (dim_v == 3) {
        zisa::shape_t<3> slice_shape(grid.N_phys, grid.N_phys, grid.N_phys);
        zisa::array_const_view<real_t, 3> u(
            slice_shape, u_host.raw(), zisa::device_type::cpu);
        zisa::array_const_view<real_t, 3> v(slice_shape,
                                            u_host.raw()
                                                + zisa::product(slice_shape),
                                            zisa::device_type::cpu);
        zisa::array_const_view<real_t, 3> w(
            slice_shape,
            u_host.raw() + 2 * zisa::product(slice_shape),
            zisa::device_type::cpu);
        zisa::save(writer, u, name + "_u");
        zisa::save(writer, v, name + "_v");
        zisa::save(writer, w, name + "_w");
      }
    }
  }
}

#if AZEBAN_HAS_MPI
template <int dim_v>
static void runFromConfig_MPI(const nlohmann::json &config, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  zisa::int_t num_samples = 1;
  if (config.contains("num_samples")) {
    num_samples = config["num_samples"];
  }

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
  auto initializer = make_initializer<dim_v>(config, rng);

  std::unique_ptr<zisa::NetCDFSerialWriter> writer;
  if (rank == 0) {
    writer = std::make_unique<zisa::NetCDFSerialWriter>(make_nc_writer<dim_v>(
        output + ".nc", num_samples, simulation, snapshots));
    zisa::save(
        *writer,
        zisa::array_const_view<real_t, 1>(zisa::shape_t<1>(snapshots.size()),
                                          snapshots.data(),
                                          zisa::device_type::cpu),
        "time");
  }

  for (zisa::int_t sample = 0; sample < num_samples; ++sample) {
    auto simulation = make_simulation_mpi<dim_v>(config, comm);
    const auto &grid = simulation.grid();

    auto u_host = grid.make_array_phys(
        simulation.n_vars(), zisa::device_type::cpu, comm);
    auto u_device = grid.make_array_phys(
        simulation.n_vars(), zisa::device_type::cuda, comm);
    auto u_hat_device = grid.make_array_fourier(
        simulation.n_vars(), zisa::device_type::cuda, comm);
    auto fft = make_fft_mpi<dim_v>(u_hat_device, u_device, comm);

    simulation.reset();
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
        const std::string name
            = "sample_" + std::to_string(sample) + "_time_" + std::to_string(t);
        if (dim_v == 1) {
          zisa::shape_t<1> slice_shape(grid.N_phys);
          zisa::array_const_view<real_t, 1> u(
              slice_shape, u_init.raw(), zisa::device_type::cpu);
          zisa::save(*writer, u, name + "_u");
        }
        if (dim_v == 2) {
          zisa::shape_t<2> slice_shape(grid.N_phys, grid.N_phys);
          zisa::array_const_view<real_t, 2> u(
              slice_shape, u_init.raw(), zisa::device_type::cpu);
          zisa::array_const_view<real_t, 2> v(slice_shape,
                                              u_init.raw()
                                                  + zisa::product(slice_shape),
                                              zisa::device_type::cpu);
          zisa::save(*writer, u, name + "_u");
          zisa::save(*writer, v, name + "_v");
        }
        if (dim_v == 3) {
          zisa::shape_t<3> slice_shape(grid.N_phys, grid.N_phys, grid.N_phys);
          zisa::array_const_view<real_t, 3> u(
              slice_shape, u_init.raw(), zisa::device_type::cpu);
          zisa::array_const_view<real_t, 3> v(slice_shape,
                                              u_init.raw()
                                                  + zisa::product(slice_shape),
                                              zisa::device_type::cpu);
          zisa::array_const_view<real_t, 3> w(
              slice_shape,
              u_init.raw() + 2 * zisa::product(slice_shape),
              zisa::device_type::cpu);
          zisa::save(*writer, u, name + "_u");
          zisa::save(*writer, v, name + "_v");
          zisa::save(*writer, w, name + "_w");
        }
      }
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
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const bool use_mpi = size > 1;
#endif

#if AZEBAN_HAS_MPI
  Profiler::start(MPI_COMM_WORLD);
#else
  Profiler::start();
#endif
#if AZEBAN_HAS_MPI
  MPI_Barrier(MPI_COMM_WORLD);
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
  Profiler::stop(MPI_COMM_WORLD);
#else
  Profiler::stop();
#endif

#if AZEBAN_DO_PROFILE
#if AZEBAN_HAS_MPI
  fmt::print("Rank {}:\n{}", rank, Profiler::summary());
  std::ofstream pstream("profiling_rank" + std::to_string(rank) + ".json");
  pstream << std::setw(2) << Profiler::json();
#else
  fmt::print(Profiler::summary());
  std::ofstream pstream("profiling.json");
  pstream << std::setw(2) << Profiler::json();
#endif
#endif

#if AZEBAN_HAS_MPI
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}
