#ifndef AZEBAN_MPI_CUDA_AWARE_COMUNICATOR_HPP_
#define AZEBAN_MPI_CUDA_AWARE_COMUNICATOR_HPP_

#include <azeban/mpi/communicator.hpp>

namespace azeban {

class CUDAAwareCommunicator : public Communicator {
  using super = Communicator;

public:
  CUDAAwareCommunicator();
  CUDAAwareCommunicator(MPI_Comm comm);
  CUDAAwareCommunicator(const CUDAAwareCommunicator &) = default;
  CUDAAwareCommunicator(CUDAAwareCommunicator &&) = default;
  virtual ~CUDAAwareCommunicator() override = default;
  CUDAAwareCommunicator &operator=(const CUDAAwareCommunicator &) = default;
  CUDAAwareCommunicator &operator=(CUDAAwareCommunicator &&) = default;

  using super::get_mpi_comm;
  using super::rank;
  using super::size;

  using super::alltoall;

protected:
  using super::comm_;

  virtual void do_alltoall(const void *sendbuf,
                           int sendcount,
                           MPI_Datatype sendtype,
                           zisa::device_type sendloc,
                           void *recvbuf,
                           int recvcount,
                           MPI_Datatype recvtype,
                           zisa::device_type recvloc) const override;
};

}

#endif
