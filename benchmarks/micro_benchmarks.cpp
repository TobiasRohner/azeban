#include <azeban/benchmark.hpp>
#include <zisa/config.hpp>
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif


class NullReporter : public benchmark::BenchmarkReporter {
public:
  NullReporter() = default;
  virtual bool ReportContext(const Context &) override { return true; }
  virtual void ReportRuns(const std::vector<Run> &) override  { }
  virtual void Finalize() override { }
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
  }
  else {
    benchmark::Initialize(&argc, argv);
    NullReporter null;
    benchmark::RunSpecifiedBenchmarks(&null);
  }

  MPI_Finalize();
#endif

  return 0;
}
