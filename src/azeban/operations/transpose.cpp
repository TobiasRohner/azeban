/* 
 * This file is part of azeban (https://github.com/TobiasRohner/azeban).
 * Copyright (c) 2021 Tobias Rohner.
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#if AZEBAN_HAS_MPI

#include <azeban/mpi_types.hpp>
#include <azeban/operations/transpose.hpp>
#include <azeban/profiler.hpp>
#include <memory>
#include <zisa/math/basic_functions.hpp>

namespace azeban {

void transpose(const zisa::array_view<complex_t, 3> &dst,
               const zisa::array_const_view<complex_t, 3> &src,
               MPI_Comm comm) {
  AZEBAN_PROFILE_START("transpose", comm);
  LOG_ERR_IF(dst.memory_location() != zisa::device_type::cpu,
             "Can only transpose data on the host");
  LOG_ERR_IF(src.memory_location() != zisa::device_type::cpu,
             "Can only transpose data on the host");

  static constexpr zisa::int_t BLOCKSIZE = 16;
  const zisa::int_t ndim = src.shape(0);
  const zisa::int_t Nx = src.shape(2);
  const zisa::int_t Ny = dst.shape(2);

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  auto N_loc = std::make_unique<zisa::int_t[]>(2 * size);
  const zisa::int_t my_N_loc[2] = {src.shape(1), dst.shape(1)};
  MPI_Allgather(my_N_loc,
                2,
                mpi_type<zisa::int_t>(),
                N_loc.get(),
                2,
                mpi_type<zisa::int_t>(),
                comm);

  auto sendbuf = std::make_unique<complex_t[]>(ndim * Nx * N_loc[2 * rank + 0]);
  auto recvbuf = std::make_unique<complex_t[]>(ndim * Ny * N_loc[2 * rank + 1]);
  auto sendcnts = std::make_unique<int[]>(size);
  auto recvcnts = std::make_unique<int[]>(size);
  auto sdispls = std::make_unique<int[]>(size + 1);
  auto rdispls = std::make_unique<int[]>(size + 1);

  // Copy the transposed data into sendbuf
  AZEBAN_PROFILE_START("transpose::preprocessing", comm);
  sdispls[0] = 0;
  rdispls[0] = 0;
  zisa::int_t i_offset = 0;
  for (int r = 0; r < size; ++r) {
    const zisa::int_t ib_end = N_loc[2 * r + 1];
    const zisa::int_t jb_end = N_loc[2 * rank + 0];
#pragma omp parallel for collapse(3)
    for (zisa::int_t d = 0; d < ndim; ++d) {
      for (zisa::int_t ib = 0; ib < ib_end; ib += BLOCKSIZE) {
        for (zisa::int_t jb = 0; jb < jb_end; jb += BLOCKSIZE) {
          const zisa::int_t i_end = zisa::min(ib + BLOCKSIZE, N_loc[2 * r + 1]);
          const zisa::int_t j_end
              = zisa::min(jb + BLOCKSIZE, N_loc[2 * rank + 0]);
          for (zisa::int_t i = ib; i < i_end; ++i) {
            for (zisa::int_t j = jb; j < j_end; ++j) {
              sendbuf[sdispls[r] + d * N_loc[2 * rank + 0] * N_loc[2 * r + 1]
                      + i * N_loc[2 * rank + 0] + j]
                  = src(d, j, i + i_offset);
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
  AZEBAN_PROFILE_STOP("transpose::preprocessing", comm);

  // Communicate with MPI_Alltoallv
  AZEBAN_PROFILE_START("transpose::communication", comm);
  MPI_Alltoallv(sendbuf.get(),
                sendcnts.get(),
                sdispls.get(),
                mpi_type<complex_t>(),
                recvbuf.get(),
                recvcnts.get(),
                rdispls.get(),
                mpi_type<complex_t>(),
                comm);
  AZEBAN_PROFILE_STOP("transpose::communication", comm);

  // Copy to dst
  AZEBAN_PROFILE_START("transpose::postprocessing", comm);
  zisa::int_t j_offset = 0;
  for (int r = 0; r < size; ++r) {
    const zisa::int_t i_end = N_loc[2 * rank + 1];
    const zisa::int_t j_end = N_loc[2 * r + 0];
#pragma omp parallel for collapse(3)
    for (zisa::int_t d = 0; d < ndim; ++d) {
      for (zisa::int_t i = 0; i < i_end; ++i) {
        for (zisa::int_t j = 0; j < j_end; ++j) {
          dst(d, i, j + j_offset)
              = recvbuf[rdispls[r] + d * N_loc[2 * rank + 1] * N_loc[2 * r + 0]
                        + i * N_loc[2 * r + 0] + j];
        }
      }
    }
    j_offset += N_loc[2 * r + 0];
  }
  AZEBAN_PROFILE_STOP("transpose::postprocessing", comm);
  AZEBAN_PROFILE_STOP("transpose", comm);
}

void transpose(const zisa::array_view<complex_t, 4> &dst,
               const zisa::array_const_view<complex_t, 4> &src,
               MPI_Comm comm) {
  AZEBAN_PROFILE_START("transpose", comm);
  LOG_ERR_IF(dst.memory_location() != zisa::device_type::cpu,
             "Can only transpose data on the host");
  LOG_ERR_IF(src.memory_location() != zisa::device_type::cpu,
             "Can only transpose data on the host");

  static constexpr zisa::int_t BLOCKSIZE = 8;
  const zisa::int_t ndim = src.shape(0);
  const zisa::int_t Nx = src.shape(2);

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  auto N_loc = std::make_unique<zisa::int_t[]>(2 * size);
  const zisa::int_t my_N_loc[2] = {src.shape(1), dst.shape(1)};
  MPI_Allgather(my_N_loc,
                2,
                mpi_type<zisa::int_t>(),
                N_loc.get(),
                2,
                mpi_type<zisa::int_t>(),
                comm);

  auto sendbuf = std::make_unique<complex_t[]>(src.size());
  auto recvbuf = std::make_unique<complex_t[]>(dst.size());
  auto sendcnts = std::make_unique<int[]>(size);
  auto recvcnts = std::make_unique<int[]>(size);
  auto sdispls = std::make_unique<int[]>(size + 1);
  auto rdispls = std::make_unique<int[]>(size + 1);

  // Copy the transposed data into sendbuf
  AZEBAN_PROFILE_START("transpose::preprocessing", comm);
  sdispls[0] = 0;
  rdispls[0] = 0;
  zisa::int_t i_offset = 0;
  for (int r = 0; r < size; ++r) {
    const zisa::int_t ib_end = N_loc[2 * r + 1];
    const zisa::int_t kb_end = N_loc[2 * rank + 0];
    const zisa::shape_t<4> view_shape(
        ndim, N_loc[2 * r + 1], Nx, N_loc[2 * rank + 0]);
    zisa::array_view<complex_t, 4> view(
        view_shape, sendbuf.get() + sdispls[r], zisa::device_type::cpu);
#pragma omp parallel for collapse(4)
    for (zisa::int_t d = 0; d < ndim; ++d) {
      for (zisa::int_t ib = 0; ib < ib_end; ib += BLOCKSIZE) {
        for (zisa::int_t jb = 0; jb < Nx; jb += BLOCKSIZE) {
          for (zisa::int_t kb = 0; kb < kb_end; kb += BLOCKSIZE) {
            const zisa::int_t i_end = zisa::min(ib + BLOCKSIZE, ib_end);
            const zisa::int_t j_end = zisa::min(jb + BLOCKSIZE, Nx);
            const zisa::int_t k_end = zisa::min(kb + BLOCKSIZE, kb_end);
            for (zisa::int_t i = ib; i < i_end; ++i) {
              for (zisa::int_t j = jb; j < j_end; ++j) {
                for (zisa::int_t k = kb; k < k_end; ++k) {
                  view(d, i, j, k) = src(d, k, j, i + i_offset);
                }
              }
            }
          }
        }
      }
    }
    i_offset += N_loc[2 * r + 1];
    sendcnts[r] = ndim * Nx * N_loc[2 * rank + 0] * N_loc[2 * r + 1];
    recvcnts[r] = ndim * Nx * N_loc[2 * r + 0] * N_loc[2 * rank + 1];
    sdispls[r + 1] = sdispls[r] + sendcnts[r];
    rdispls[r + 1] = rdispls[r] + recvcnts[r];
  }
  AZEBAN_PROFILE_STOP("transpose::preprocessing", comm);

  // Communicate with MPI_Alltoallv
  AZEBAN_PROFILE_START("transpose::communication", comm);
  MPI_Alltoallv(sendbuf.get(),
                sendcnts.get(),
                sdispls.get(),
                mpi_type<complex_t>(),
                recvbuf.get(),
                recvcnts.get(),
                rdispls.get(),
                mpi_type<complex_t>(),
                comm);
  AZEBAN_PROFILE_STOP("transpose::communication", comm);

  // Copy to dst
  AZEBAN_PROFILE_START("transpose::postprocessing", comm);
  zisa::int_t k_offset = 0;
  for (int r = 0; r < size; ++r) {
    const zisa::int_t i_end = N_loc[2 * rank + 1];
    const zisa::int_t k_end = N_loc[2 * r + 0];
    const zisa::shape_t<4> view_shape(
        ndim, N_loc[2 * rank + 1], Nx, N_loc[2 * r + 0]);
    zisa::array_view<complex_t, 4> view(
        view_shape, recvbuf.get() + rdispls[r], zisa::device_type::cpu);
#pragma omp parallel for collapse(4)
    for (zisa::int_t d = 0; d < ndim; ++d) {
      for (zisa::int_t i = 0; i < i_end; ++i) {
        for (zisa::int_t j = 0; j < Nx; ++j) {
          for (zisa::int_t k = 0; k < k_end; ++k) {
            dst(d, i, j, k + k_offset) = view(d, i, j, k);
          }
        }
      }
    }
    k_offset += N_loc[2 * r + 0];
  }
  AZEBAN_PROFILE_STOP("transpose::postprocessing", comm);
  AZEBAN_PROFILE_STOP("transpose", comm);
}

}

#endif
