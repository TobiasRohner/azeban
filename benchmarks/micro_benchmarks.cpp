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
#include <azeban/benchmark.hpp>
#include <zisa/config.hpp>
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif

class NullReporter : public benchmark::BenchmarkReporter {
public:
  NullReporter() = default;
  virtual bool ReportContext(const Context &) override { return true; }
  virtual void ReportRuns(const std::vector<Run> &) override {}
  virtual void Finalize() override {}
};

int main(int argc, char *argv[]) {
#if AZEBAN_HAS_MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  LOG_ERR_IF(provided < MPI_THREAD_FUNNELED,
             "MPI did not provide enough thread safety");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
#endif
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
#if AZEBAN_HAS_MPI
  } else {
    benchmark::Initialize(&argc, argv);
    NullReporter null;
    benchmark::RunSpecifiedBenchmarks(&null);
  }

  MPI_Finalize();
#endif

  return 0;
}
