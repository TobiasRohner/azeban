#if AZEBAN_HAS_MPI

#include <azeban/operations/transpose.hpp>
#include <azeban/mpi_types.hpp>
#include <azeban/profiler.hpp>
#include <zisa/math/basic_functions.hpp>
#include <memory>



namespace azeban {

void transpose(const zisa::array_view<complex_t, 3> &dst, const zisa::array_const_view<complex_t, 3> &src, MPI_Comm comm) {
  AZEBAN_PROFILE_START("transpose", comm);
  LOG_ERR_IF(dst.memory_location() != zisa::device_type::cpu, "Can only transpose data on the host");
  LOG_ERR_IF(src.memory_location() != zisa::device_type::cpu, "Can only transpose data on the host");

  static constexpr zisa::int_t BLOCKSIZE = 16;
  const zisa::int_t ndim = src.shape(0);
  const zisa::int_t Nx = src.shape(2);
  const zisa::int_t Ny = dst.shape(2);

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  auto N_loc = std::make_unique<zisa::int_t[]>(2 * size);
  const zisa::int_t my_N_loc[2] = {src.shape(1), dst.shape(1)};
  MPI_Allgather(my_N_loc, 2, mpi_type<zisa::int_t>(), N_loc.get(), 2, mpi_type<zisa::int_t>(), comm);

  auto sendbuf = std::make_unique<complex_t[]>(ndim * Nx * N_loc[2 * rank + 0]);
  auto recvbuf = std::make_unique<complex_t[]>(ndim * Ny * N_loc[2 * rank + 1]);
  auto sendcnts = std::make_unique<int[]>(size);
  auto recvcnts = std::make_unique<int[]>(size);
  auto sdispls = std::make_unique<int[]>(size + 1);
  auto rdispls = std::make_unique<int[]>(size + 1);

  // Copy the transposed data into sendbuf
  sdispls[0] = 0;
  rdispls[0] = 0;
  zisa::int_t i_offset = 0;
  for (int r = 0 ; r < size ; ++r) {
    for (zisa::int_t d = 0 ; d < ndim ; ++d) {
      for (zisa::int_t ib = 0 ; ib < N_loc[2 * r + 1] ; ib += BLOCKSIZE) {
	for (zisa::int_t jb = 0 ; jb < N_loc[2 * rank + 0] ; jb += BLOCKSIZE) {
	  const zisa::int_t i_end = zisa::min(ib + BLOCKSIZE, N_loc[2 * r + 1]);
	  const zisa::int_t j_end = zisa::min(jb + BLOCKSIZE, N_loc[2 * rank + 0]);
	  for (zisa::int_t i = ib ; i < i_end ; ++i) {
	    for (zisa::int_t j = jb ; j < j_end ; ++j) {
	      sendbuf[sdispls[r] + d * N_loc[2 * rank + 0] * N_loc[2 * r + 1] + i * N_loc[2 * rank + 0] + j] = src(d, j, i + i_offset);
	    }
	  }
	}
      }
    }
    i_offset += N_loc[2 * r + 1];
    sendcnts[r] = ndim * N_loc[2 * rank + 0] * N_loc[2 * r + 1];
    recvcnts[r] = ndim * N_loc[2 * r + 0] * N_loc[2 * rank + 1];
    sdispls[r + 1] = sdispls[r] + sendcnts[r];
    rdispls[r + 1] = rdispls[r] + recvcnts[r];
  }

  // Communicate with MPI_Alltoallv
  MPI_Alltoallv(sendbuf.get(),
		sendcnts.get(),
		sdispls.get(),
		mpi_type<complex_t>(),
		recvbuf.get(),
		recvcnts.get(),
		rdispls.get(),
		mpi_type<complex_t>(),
		comm);

  // Copy to dst
  zisa::int_t j_offset = 0;
  for (int r = 0 ; r < size ; ++r) {
    for (zisa::int_t d = 0 ; d < ndim ; ++d) {
      for (zisa::int_t i = 0 ; i < N_loc[2 * rank + 1] ; ++i) {
	for (zisa::int_t j = 0 ; j < N_loc[2 * r + 0] ; ++j) {
	  dst(d, i, j + j_offset) = recvbuf[rdispls[r] + d * N_loc[2 * rank + 1] * N_loc[2 * r + 0] + i * N_loc[2 * r + 0] + j];
	}
      }
    }
    j_offset += N_loc[2 * r + 0];
  }
  AZEBAN_PROFILE_STOP("transpose", comm);
}

}

#endif
