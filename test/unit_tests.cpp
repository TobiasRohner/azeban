#define CATCH_CONFIG_RUNNER
#include <azeban/catch.hpp>
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif
#include <zisa/config.hpp>

int main(int argc, char *argv[]) {
#if AZEBAN_HAS_MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  LOG_ERR_IF(provided < MPI_THREAD_FUNNELED,
             "MPI did not provide enough thread safety");
#endif
  int result = Catch::Session().run(argc, argv);
#if AZEBAN_HAS_MPI
  MPI_Finalize();
#endif
  return result;
}
