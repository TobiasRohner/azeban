#include <azeban/mpi/cuda_aware_communicator.hpp>

#if AZEBAN_HAS_CUDA_AWARE_MPI
#if MPICH
#define CUDA_ENABLE_ENV_VAR "MPICH_RDMA_ENABLED_CUDA"
#define EXPECTED_ENV_VAR_VALUE "1"
#elif OPEN_MPI
#define CUDA_ENABLE_ENV_VAR "OMPI_MCA_opal_cuda_support"
#define EXPECTED_ENV_VAR_VALUE "true"
#else
#warning                                                                       \
    "AZEBAN_HAS_CUDA_AWARE_MPI set but compiled with unsupported MPI Implementation"
#endif
#endif

namespace azeban {

CUDAAwareCommunicator::CUDAAwareCommunicator()
    : CUDAAwareCommunicator(MPI_COMM_WORLD) {}

CUDAAwareCommunicator::CUDAAwareCommunicator(MPI_Comm comm) : super(comm) {
#ifdef CUDA_ENABLE_ENV_VAR
  const char *cuda_enable_env_var = getenv(CUDA_ENABLE_ENV_VAR);
  bool is_cuda_aware = false;
  if (cuda_enable_env_var) {
    is_cuda_aware = strcmp(cuda_enable_env_var, EXPECTED_ENV_VAR_VALUE) == 0;
  }
  if (!is_cuda_aware) {
    const std::string err_msg = fmt::format(
        "Runtime was unable to detect MPI cuda awarenes.\nPlease try running "
        "again with the evironment variable {} set to {}.\n",
        CUDA_ENABLE_ENV_VAR,
        EXPECTED_ENV_VAR_VALUE);
    LOG_ERR(err_msg);
  }
#else
  LOG_ERR("CUDAAwareCommunicator is unavailable. Please recompile with "
          "ENABLE_CUDA_AWARE_MPI=ON");
#endif
}

void CUDAAwareCommunicator::do_alltoall(const void *sendbuf,
                                        int sendcount,
                                        MPI_Datatype sendtype,
                                        zisa::device_type sendloc,
                                        void *recvbuf,
                                        int recvcount,
                                        MPI_Datatype recvtype,
                                        zisa::device_type recvloc) const {
  MPI_Alltoall(
      sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm_);
}

}

#undef CUDA_ENABLE_ENV_VAR
#undef EXPECTED_ENV_VAR_VALUE
