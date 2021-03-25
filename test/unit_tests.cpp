#define CATCH_CONFIG_RUNNER
#include <azeban/catch.hpp>
#include <mpi.h>
#include <zisa/config.hpp>

int main(int argc, char *argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  LOG_ERR_IF(provided < MPI_THREAD_FUNNELED,
             "MPI did not provide enough thread safety");
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
