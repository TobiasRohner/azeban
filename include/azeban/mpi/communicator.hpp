#ifndef AZEBAN_MPI_COMMUNICATOR_HPP_
#define AZEBAN_MPI_COMMUNICATOR_HPP_

#include <azeban/mpi/mpi_types.hpp>
#include <mpi.h>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>
#include <zisa/utils/logging.hpp>

namespace azeban {

class Communicator {
public:
  Communicator();
  Communicator(MPI_Comm comm);
  Communicator(const Communicator &) = default;
  Communicator(Communicator &&) = default;
  virtual ~Communicator() = default;
  Communicator &operator=(const Communicator &) = default;
  Communicator &operator=(Communicator &&) = default;

  int size() const;
  int rank() const;
  MPI_Comm get_mpi_comm() const;

  template <typename TS, typename TR, int DimS, int DimR>
  void alltoall(const zisa::array_const_view<TS, DimS> &sendbuf,
                const zisa::array_view<TR, DimR> &recvbuf) const {
    const int sendcount = sendbuf.size() / size();
    const int recvcount = recvbuf.size() / size();
    do_alltoall(sendbuf.raw(),
                sendcount,
                mpi_type<TS>(),
                sendbuf.memory_location(),
                recvbuf.raw(),
                recvcount,
                mpi_type<TR>(),
                recvbuf.memory_location());
  }

  template <typename TS, typename TR, int DimS, int DimR>
  void alltoall(const zisa::array_view<TS, DimS> &sendbuf,
                const zisa::array_view<TR, DimR> &recvbuf) const {
    alltoall(zisa::array_const_view<TS, DimS>(sendbuf), recvbuf);
  }

protected:
  MPI_Comm comm_;

  virtual void do_alltoall(const void *sendbuf,
                           int sendcount,
                           MPI_Datatype sendtype,
                           zisa::device_type sendloc,
                           void *recvbuf,
                           int recvcount,
                           MPI_Datatype recvtype,
                           zisa::device_type recvloc) const;
};

}

#endif
