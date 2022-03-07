#ifndef AZEBAN_MPI_MANUAL_CUDA_AWARE_COMMUNICATOR_HPP_
#define AZEBAN_MPI_MANUAL_CUDA_AWARE_COMMUNICATOR_HPP_

#include <azeban/mpi/communicator.hpp>

namespace azeban {

class ManualCUDAAwareCommunicator : public Communicator {
  using super = Communicator;

public:
  ManualCUDAAwareCommunicator();
  ManualCUDAAwareCommunicator(MPI_Comm comm);
  ManualCUDAAwareCommunicator(const ManualCUDAAwareCommunicator &) = default;
  ManualCUDAAwareCommunicator(ManualCUDAAwareCommunicator &&) = default;
  virtual ~ManualCUDAAwareCommunicator() override = default;
  ManualCUDAAwareCommunicator &operator=(const ManualCUDAAwareCommunicator &)
      = default;
  ManualCUDAAwareCommunicator &operator=(ManualCUDAAwareCommunicator &&)
      = default;

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
