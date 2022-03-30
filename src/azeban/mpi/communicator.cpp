#if AZEBAN_HAS_MPI

#include <azeban/mpi/communicator.hpp>
#include <cstdlib>
#include <fmt/core.h>

namespace azeban {

Communicator::Communicator() : Communicator(MPI_COMM_WORLD) {}

Communicator::Communicator(MPI_Comm comm) : comm_(comm) {}

int Communicator::size() const {
  int s;
  MPI_Comm_size(comm_, &s);
  return s;
}

int Communicator::rank() const {
  int r;
  MPI_Comm_rank(comm_, &r);
  return r;
}

MPI_Comm Communicator::get_mpi_comm() const { return comm_; }

void Communicator::do_alltoall(const void *sendbuf,
                               int sendcount,
                               MPI_Datatype sendtype,
                               zisa::device_type sendloc,
                               void *recvbuf,
                               int recvcount,
                               MPI_Datatype recvtype,
                               zisa::device_type recvloc) const {
  LOG_ERR_IF(sendloc != zisa::device_type::cpu,
             "Expected sendbuf to be on the host");
  LOG_ERR_IF(recvloc != zisa::device_type::cpu,
             "Expected recvbuf to be on the host");
  MPI_Alltoall(
      sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm_);
}

}

#endif
