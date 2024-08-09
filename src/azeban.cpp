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
#include <boost/program_options.hpp>
#include <fmt/core.h>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/manual_cuda_aware_communicator.hpp>
#endif

using namespace azeban;
namespace po = boost::program_options;

nlohmann::json read_config(const std::string &config_filename) {
  // Ensures the file `config_filename` is closed again as soon as possible.
  std::ifstream config_file(config_filename);
  nlohmann::json config;
  config_file >> config;

  return config;
}

void run_azeban(const nlohmann::json &config,
                zisa::int_t total_samples,
                const std::string &original_config) {
  fmt::print("Running azeban in single-node mode.\nRun cofiguration is\n{}\n",
             config.dump(2));
  run_from_config(config, total_samples, original_config);
}

#if AZEBAN_HAS_MPI
void run_azeban(const nlohmann::json &config,
                zisa::int_t total_samples,
                const std::string &original_config,
                MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  if (size > 1) {
    if (rank == 0) {
      fmt::print(
          "Running azeban with {} MPI ranks.\nRun configuration is\n{}\n",
          size,
          config.dump(2));
    }
    Communicator communicator(comm);
    run_from_config(config, total_samples, original_config, &communicator);
  } else {
    run_azeban(config, total_samples, original_config);
  }
}
#endif

int main(int argc, char *argv[]) {
#if AZEBAN_HAS_MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  LOG_ERR_IF(provided < MPI_THREAD_FUNNELED,
             "MPI did not provide enough thread safety");
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

  int ranks_per_sample;
  std::string config_path;
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "config",
      po::value<std::string>(&config_path),
      "Path to the simulation config file")(
      "ranks-per-sample",
      po::value<int>(&ranks_per_sample)->default_value(1),
      "How many MPI ranks to use for a single sample");
  po::positional_options_description pdesc;
  pdesc.add("config", -1);
  po::variables_map vm;
  po::store(
      po::command_line_parser(argc, argv).options(desc).positional(pdesc).run(),
      vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    exit(1);
  }
  if (config_path == "") {
    std::cout << "ERROR: No config path provided" << std::endl;
    std::cout << desc << std::endl;
    exit(1);
  }
#if AZEBAN_HAS_MPI
  if (size % ranks_per_sample) {
    std::cout
        << "ERROR: ranks-per-sample must divide the total number of MPI ranks"
        << std::endl;
    exit(1);
  }
#endif

  auto config = read_config(config_path);
  const std::string original_config = config.dump(2, ' ', true);

  Profiler::start();
  int num_samples = 1;
  if (config.contains("num_samples")) {
    num_samples = config["num_samples"];
  }
#if AZEBAN_HAS_MPI
  const int color = rank / ranks_per_sample;
  MPI_Comm subcomm;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &subcomm);
  int sample_idx_start = 0;
  if (config.contains("sample_idx_start")) {
    sample_idx_start = config["sample_idx_start"];
  }
  size_t seed = 1;
  if (config.contains("seed")) {
    seed = config["seed"];
  }
  const int num_par_samples = size / ranks_per_sample;
  const int samples_per_comm = num_samples / num_par_samples;
  config["seed"] = seed + color;
  config["num_samples"] = samples_per_comm;
  config["sample_idx_start"] = sample_idx_start + color * samples_per_comm;
  run_azeban(config, num_samples, original_config, subcomm);
#else
  run_azeban(config, num_samples, original_config);
#endif
  Profiler::stop();

#if AZEBAN_DO_PROFILE
#if AZEBAN_HAS_MPI
  for (int r = 0; r < size; ++r) {
    if (r == rank) {
      std::cout << "Rank " << rank << ":\n";
      Profiler::summarize(std::cout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  std::ofstream pstream("profiling_rank" + std::to_string(rank) + ".out");
  Profiler::serialize(pstream);
#else
  Profiler::summarize(std::cout);
  std::ofstream pstream("profiling.out");
  Profiler::serialize(pstream);
#endif
#endif

#if AZEBAN_HAS_MPI
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}
