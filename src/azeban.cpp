/*
 * This file is part of azeban (https://github.com/TobiasRohner/azeban).
 * Copyright (c) 2021 Tobias Rohner.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include <azeban/init/initializer_factory.hpp>
#include <azeban/operations/fft.hpp>
#include <azeban/profiler.hpp>
#include <azeban/sequence_factory.hpp>
#include <azeban/simulation_factory.hpp>
#include <cstdlib>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fstream>
#include <hdf5.h>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <random>
#include <zisa/config.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>
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
                                        const Simulation<Dim> &simulation) {
  const Grid<Dim> &grid = simulation.grid();

  using nc_dim_t = std::tuple<std::string, size_t>;
  using nc_var_t
      = std::tuple<std::string, std::vector<std::string>, zisa::ErasedDataType>;
  std::vector<nc_dim_t> dims;
  std::vector<nc_var_t> vars;

  dims.emplace_back("N", grid.N_phys);

  std::vector<std::string> vardim(Dim, "N");

  vars.emplace_back("u", vardim, zisa::erase_data_type<real_t>());
  if (Dim > 1) {
    vars.emplace_back("v", vardim, zisa::erase_data_type<real_t>());
  }
  if (Dim > 2) {
    vars.emplace_back("w", vardim, zisa::erase_data_type<real_t>());
  }
  if (simulation.n_vars() == Dim + 1) {
    vars.emplace_back("rho", vardim, zisa::erase_data_type<real_t>());
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
  const auto &grid = simulation.grid();
  auto initializer = make_initializer<dim_v>(config, rng);

  std::filesystem::path sample_folder = output;
  if (!std::filesystem::exists(sample_folder)) {
    std::filesystem::create_directories(sample_folder);
  }

  for (zisa::int_t sample = sample_idx_start;
       sample < sample_idx_start + num_samples;
       ++sample) {
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
      const std::string name = "sample_" + std::to_string(sample) + "_time_"
                               + std::to_string(t + time_offset) + ".nc";
      auto writer = make_nc_writer<dim_v>(output + "/" + name, simulation);
      if (dim_v == 1) {
        zisa::shape_t<1> slice_shape(grid.N_phys);
        zisa::array_const_view<real_t, 1> u(
            slice_shape, u_host.raw(), zisa::device_type::cpu);
        zisa::save(writer, u, "u");
        if (simulation.n_vars() == 2) {
          zisa::array_const_view<real_t, 1> rho(
              slice_shape,
              u_host.raw() + zisa::product(slice_shape),
              zisa::device_type::cpu);
          zisa::save(writer, rho, "rho");
        }
      }
      if (dim_v == 2) {
        zisa::shape_t<2> slice_shape(grid.N_phys, grid.N_phys);
        zisa::array_const_view<real_t, 2> u(
            slice_shape, u_host.raw(), zisa::device_type::cpu);
        zisa::array_const_view<real_t, 2> v(slice_shape,
                                            u_host.raw()
                                                + zisa::product(slice_shape),
                                            zisa::device_type::cpu);
        zisa::save(writer, u, "u");
        zisa::save(writer, v, "v");
        if (simulation.n_vars() == 3) {
          zisa::array_const_view<real_t, 2> rho(
              slice_shape,
              u_host.raw() + 2 * zisa::product(slice_shape),
              zisa::device_type::cpu);
          zisa::save(writer, rho, "rho");
        }
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
        zisa::save(writer, u, "u");
        zisa::save(writer, v, "v");
        zisa::save(writer, w, "w");
        if (simulation.n_vars() == 4) {
          zisa::array_const_view<real_t, 3> rho(
              slice_shape,
              u_host.raw() + 3 * zisa::product(slice_shape),
              zisa::device_type::cpu);
          zisa::save(writer, rho, "rho");
        }
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
  auto initializer = make_initializer<dim_v>(config, rng);

  if (rank == 0) {
    std::filesystem::path sample_folder = output;
    if (!std::filesystem::exists(sample_folder)) {
      std::filesystem::create_directories(sample_folder);
    }
  }

  for (zisa::int_t sample = sample_idx_start;
       sample < sample_idx_start + num_samples;
       ++sample) {
    simulation = make_simulation_mpi<dim_v>(config, comm);
    const auto &grid = simulation.grid();

    auto u_host = grid.make_array_phys(
        simulation.n_vars(), zisa::device_type::cpu, comm);
    auto u_device = grid.make_array_phys(
        simulation.n_vars(), zisa::device_type::cuda, comm);
    auto u_hat_device = grid.make_array_fourier(
        simulation.n_vars(), zisa::device_type::cuda, comm);
    auto fft = make_fft_mpi<dim_v>(u_hat_device,
                                   u_device,
                                   comm,
                                   FFT_FORWARD | FFT_BACKWARD,
                                   simulation.equation()->get_fft_work_area());

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
        fmt::print("Sample: {}, Time: {}\n", sample, t);
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
        const std::string name = "sample_" + std::to_string(sample) + "_time_"
                                 + std::to_string(t + time_offset) + ".nc";
        auto writer = make_nc_writer<dim_v>(output + "/" + name, simulation);
        if (dim_v == 1) {
          zisa::shape_t<1> slice_shape(grid.N_phys);
          zisa::array_const_view<real_t, 1> u(
              slice_shape, u_init.raw(), zisa::device_type::cpu);
          zisa::save(writer, u, "u");
          if (simulation.n_vars() == 2) {
            zisa::array_const_view<real_t, 1> rho(
                slice_shape,
                u_init.raw() + zisa::product(slice_shape),
                zisa::device_type::cpu);
            zisa::save(writer, rho, "rho");
          }
        }
        if (dim_v == 2) {
          zisa::shape_t<2> slice_shape(grid.N_phys, grid.N_phys);
          zisa::array_const_view<real_t, 2> u(
              slice_shape, u_init.raw(), zisa::device_type::cpu);
          zisa::array_const_view<real_t, 2> v(slice_shape,
                                              u_init.raw()
                                                  + zisa::product(slice_shape),
                                              zisa::device_type::cpu);
          zisa::save(writer, u, "u");
          zisa::save(writer, v, "v");
          if (simulation.n_vars() == 3) {
            zisa::array_const_view<real_t, 2> rho(
                slice_shape,
                u_init.raw() + 2 * zisa::product(slice_shape),
                zisa::device_type::cpu);
            zisa::save(writer, rho, "rho");
          }
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
          zisa::save(writer, u, "u");
          zisa::save(writer, v, "v");
          zisa::save(writer, w, "w");
          if (simulation.n_vars() == 4) {
            zisa::array_const_view<real_t, 3> rho(
                slice_shape,
                u_init.raw() + 3 * zisa::product(slice_shape),
                zisa::device_type::cpu);
            zisa::save(writer, rho, "rho");
          }
        }
      }
    }
  }
}
#endif

nlohmann::json read_config(const std::string &config_filename) {
  // Ensures the file `config_filename` is closed again as soon as possible.
  std::ifstream config_file(config_filename);
  nlohmann::json config;
  config_file >> config;

  return config;
}

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

  auto config = read_config(argv[1]);

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
