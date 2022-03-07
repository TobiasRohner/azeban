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
#include <azeban/profiler.hpp>
#include <azeban/run_from_config.hpp>
#include <fmt/core.h>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/manual_cuda_aware_communicator.hpp>
#endif

using namespace azeban;

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
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const bool use_mpi = size > 1;
#endif

  if (argc != 2) {
    fmt::print(stderr, "Usage: {} <config>\n", argv[0]);
    exit(1);
  }

  auto config = read_config(argv[1]);

#if AZEBAN_HAS_MPI
  if (size > 1) {
    if (rank == 0) {
      fmt::print(
          "Running azeban with {} MPI ranks.\nRun configuration is\n{}\n",
          size,
          config.dump(2));
    }
  } else
#endif
    fmt::print("Running azeban in single-node mode.\nRun cofiguration is\n{}\n",
               config.dump(2));

#if AZEBAN_HAS_MPI
  Profiler::start(MPI_COMM_WORLD);
#else
  Profiler::start();
#endif
#if AZEBAN_HAS_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  if (use_mpi) {
    ManualCUDAAwareCommunicator comm(MPI_COMM_WORLD);
    run_from_config(config, &comm);
  } else {
#endif
    run_from_config(config);
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
