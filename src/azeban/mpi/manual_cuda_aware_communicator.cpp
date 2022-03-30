#if AZEBAN_HAS_MPI

#include <azeban/mpi/manual_cuda_aware_communicator.hpp>
#if ZISA_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace azeban {

namespace internal {

MPI_Aint buffer_size(MPI_Datatype type, int count) {
  MPI_Aint lb, extent, true_lb, true_extent;
  MPI_Type_get_extent(type, &lb, &extent);
  MPI_Type_get_true_extent(type, &true_lb, &true_extent);
  return (count - 1) * extent + true_extent;
}

template <zisa::device_type DS, zisa::device_type DR>
void alltoall(const void *sendbuf,
              int sendcount,
              MPI_Datatype sendtype,
              void *recvbuf,
              int recvcount,
              MPI_Datatype recvtype,
              MPI_Comm comm);

template <>
void alltoall<zisa::device_type::cpu, zisa::device_type::cpu>(
    const void *sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    MPI_Comm comm) {
  MPI_Alltoall(
      sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
}

#if ZISA_HAS_CUDA
template <>
void alltoall<zisa::device_type::cpu, zisa::device_type::cuda>(
    const void *sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  const MPI_Aint recvbuf_size = size * buffer_size(recvtype, recvcount);
  void *recvbuf_cpu = malloc(recvbuf_size);
  MPI_Alltoall(
      sendbuf, sendcount, sendtype, recvbuf_cpu, recvcount, recvtype, comm);
  cudaMemcpy(recvbuf, recvbuf_cpu, recvbuf_size, cudaMemcpyHostToDevice);
  free(recvbuf_cpu);
}

template <>
void alltoall<zisa::device_type::cuda, zisa::device_type::cpu>(
    const void *sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  const MPI_Aint sendbuf_size = size * buffer_size(sendtype, sendcount);
  void *sendbuf_cpu = malloc(sendbuf_size);
  cudaMemcpy(sendbuf_cpu, sendbuf, sendbuf_size, cudaMemcpyDeviceToHost);
  MPI_Alltoall(
      sendbuf_cpu, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  free(sendbuf_cpu);
}

template <>
void alltoall<zisa::device_type::cuda, zisa::device_type::cuda>(
    const void *sendbuf,
    int sendcount,
    MPI_Datatype sendtype,
    void *recvbuf,
    int recvcount,
    MPI_Datatype recvtype,
    MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  const MPI_Aint sendbuf_size = size * buffer_size(sendtype, sendcount);
  const MPI_Aint recvbuf_size = size * buffer_size(recvtype, recvcount);
  void *sendbuf_cpu = malloc(sendbuf_size);
  void *recvbuf_cpu = malloc(recvbuf_size);
  cudaMemcpy(sendbuf_cpu, sendbuf, sendbuf_size, cudaMemcpyDeviceToHost);
  MPI_Alltoall(
      sendbuf_cpu, sendcount, sendtype, recvbuf_cpu, recvcount, recvtype, comm);
  cudaMemcpy(recvbuf, recvbuf_cpu, recvbuf_size, cudaMemcpyHostToDevice);
  free(recvbuf_cpu);
  free(sendbuf_cpu);
}
#endif

}

ManualCUDAAwareCommunicator::ManualCUDAAwareCommunicator()
    : ManualCUDAAwareCommunicator(MPI_COMM_WORLD) {}

ManualCUDAAwareCommunicator::ManualCUDAAwareCommunicator(MPI_Comm comm)
    : super(comm) {}

void ManualCUDAAwareCommunicator::do_alltoall(const void *sendbuf,
                                              int sendcount,
                                              MPI_Datatype sendtype,
                                              zisa::device_type sendloc,
                                              void *recvbuf,
                                              int recvcount,
                                              MPI_Datatype recvtype,
                                              zisa::device_type recvloc) const {
  if (sendloc == zisa::device_type::cpu && recvloc == zisa::device_type::cpu) {
    internal::alltoall<zisa::device_type::cpu, zisa::device_type::cpu>(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm_);
  }
#if ZISA_HAS_CUDA
  else if (sendloc == zisa::device_type::cpu
           && recvloc == zisa::device_type::cuda) {
    internal::alltoall<zisa::device_type::cpu, zisa::device_type::cuda>(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm_);
  } else if (sendloc == zisa::device_type::cuda
             && recvloc == zisa::device_type::cpu) {
    internal::alltoall<zisa::device_type::cuda, zisa::device_type::cpu>(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm_);
  } else if (sendloc == zisa::device_type::cuda
             && recvloc == zisa::device_type::cuda) {
    internal::alltoall<zisa::device_type::cuda, zisa::device_type::cuda>(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm_);
  }
#endif
  else {
    LOG_ERR("Unsupported combination of memory locations");
  }
}

}

#endif
