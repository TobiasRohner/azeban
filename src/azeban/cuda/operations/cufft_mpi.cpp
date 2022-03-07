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
#include <azeban/cuda/operations/cufft_mpi.hpp>
#include <azeban/mpi/mpi_types.hpp>
#include <azeban/operations/transpose.hpp>
#include <azeban/profiler.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>

namespace azeban {

CUFFT_MPI<2>::CUFFT_MPI(const zisa::array_view<complex_t, 3> &u_hat,
                        const zisa::array_view<real_t, 3> &u,
                        const Communicator *comm,
                        int direction,
                        void *work_area)
    : super(direction),
      comm_(comm),
      work_area_(work_area),
      free_work_area_(work_area == nullptr) {
  initialize(u_hat, u);
}

CUFFT_MPI<2>::~CUFFT_MPI() {
  if (direction_ & FFT_FORWARD) {
    cufftDestroy(plan_forward_r2c_);
    cufftDestroy(plan_forward_c2c_);
  }
  if (direction_ & FFT_BACKWARD) {
    cufftDestroy(plan_backward_c2r_);
    cufftDestroy(plan_backward_c2c_);
  }
  if (free_work_area_) {
    cudaFree(work_area_);
  }
}

size_t CUFFT_MPI<2>::get_work_area_size() const { return 0; }

void CUFFT_MPI<2>::forward() {
  LOG_ERR_IF((direction_ & FFT_FORWARD) == 0,
             "Forward operation was not initialized");
  AZEBAN_PROFILE_START("CUFFT_MPI::forward", comm_->get_mpi_comm());
  // TODO: Remove the C-style casts when the compiler chooses not to ignore the
  // `constexpr` anymore
  if constexpr (std::is_same_v<float, real_t>) {
    // Perform the local FFTs
    cufftResult status
        = cufftExecR2C(plan_forward_r2c_,
                       (float *)u_.raw(),
                       reinterpret_cast<cufftComplex *>(partial_u_hat_.raw()));
    cudaCheckError(status);
    cudaDeviceSynchronize();
    // Transpose the data from partial_u_hat_ to u_hat_
    AZEBAN_PROFILE_START("CUFFT_MPI::forward::transpose",
                         comm_->get_mpi_comm());
    zisa::copy(mpi_send_buffer_, partial_u_hat_);
    transpose(mpi_recv_buffer_, mpi_send_buffer_, comm_);
    zisa::copy(u_hat_, mpi_recv_buffer_);
    AZEBAN_PROFILE_STOP("CUFFT_MPI::forward::transpose", comm_->get_mpi_comm());
    // Perform the final local FFTs in place
    status = cufftExecC2C(plan_forward_c2c_,
                          reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                          reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                          CUFFT_FORWARD);
    cudaCheckError(status);
    cudaDeviceSynchronize();
  } else {
    // Perform the local FFTs
    cufftResult status = cufftExecD2Z(
        plan_forward_r2c_,
        (double *)u_.raw(),
        reinterpret_cast<cufftDoubleComplex *>(partial_u_hat_.raw()));
    cudaCheckError(status);
    cudaDeviceSynchronize();
    // Transpose the data from partial_u_hat_ to u_hat_
    AZEBAN_PROFILE_START("CUFFT_MPI::forward::transpose",
                         comm_->get_mpi_comm());
    zisa::copy(mpi_send_buffer_, partial_u_hat_);
    transpose(mpi_recv_buffer_, mpi_send_buffer_, comm_);
    zisa::copy(u_hat_, mpi_recv_buffer_);
    AZEBAN_PROFILE_STOP("CUFFT_MPI::forward::transpose", comm_->get_mpi_comm());
    // Perform the final local FFTs in place
    status = cufftExecZ2Z(plan_forward_c2c_,
                          reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                          reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                          CUFFT_FORWARD);
    cudaCheckError(status);
    cudaDeviceSynchronize();
  }
  AZEBAN_PROFILE_STOP("CUFFT_MPI::forward", comm_->get_mpi_comm());
}

void *CUFFT_MPI<2>::get_work_area() const { return work_area_; }

void CUFFT_MPI<2>::backward() {
  LOG_ERR_IF((direction_ & FFT_BACKWARD) == 0,
             "Backward operation was not initialized");
  AZEBAN_PROFILE_START("CUFFT_MPI::backward", comm_->get_mpi_comm());
  // TODO: Remove the C-style casts when the compiler chooses not to ignore the
  // `constexpr` anymore
  if constexpr (std::is_same_v<float, real_t>) {
    // Perform the local FFTs
    cufftResult status
        = cufftExecC2C(plan_backward_c2c_,
                       reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                       reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                       CUFFT_INVERSE);
    cudaCheckError(status);
    cudaDeviceSynchronize();
    // Transpose the data from partial_u_hat_ to u_hat_
    AZEBAN_PROFILE_START("CUFFT_MPI::backward::transpose",
                         comm_->get_mpi_comm());
    zisa::copy(mpi_recv_buffer_, u_hat_);
    transpose(mpi_send_buffer_, mpi_recv_buffer_, comm_);
    zisa::copy(partial_u_hat_, mpi_send_buffer_);
    AZEBAN_PROFILE_STOP("CUFFT_MPI::backward::transpose",
                        comm_->get_mpi_comm());
    // Perform the final local FFTs in place
    status
        = cufftExecC2R(plan_backward_c2r_,
                       reinterpret_cast<cufftComplex *>(partial_u_hat_.raw()),
                       (float *)u_.raw());
    cudaCheckError(status);
    cudaDeviceSynchronize();
  } else {
    // Perform the local FFTs
    cufftResult status
        = cufftExecZ2Z(plan_backward_c2c_,
                       reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                       reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                       CUFFT_INVERSE);
    cudaCheckError(status);
    cudaDeviceSynchronize();
    // Transpose the data from partial_u_hat_ to u_hat_
    AZEBAN_PROFILE_START("CUFFT_MPI::backward::transpose",
                         comm_->get_mpi_comm());
    zisa::copy(mpi_recv_buffer_, u_hat_);
    transpose(mpi_send_buffer_, mpi_recv_buffer_, comm_);
    zisa::copy(partial_u_hat_, mpi_send_buffer_);
    AZEBAN_PROFILE_STOP("CUFFT_MPI::backward::transpose",
                        comm_->get_mpi_comm());
    // Perform the final local FFTs in place
    status = cufftExecZ2D(
        plan_backward_c2r_,
        reinterpret_cast<cufftDoubleComplex *>(partial_u_hat_.raw()),
        (double *)u_.raw());
    cudaCheckError(status);
    cudaDeviceSynchronize();
  }
  AZEBAN_PROFILE_STOP("CUFFT_MPI::backward", comm_->get_mpi_comm());
}

void CUFFT_MPI<2>::do_initialize(const zisa::array_view<complex_t, 3> &u_hat,
                                 const zisa::array_view<real_t, 3> &u,
                                 bool allocate_work_area) {
  ZISA_UNUSED(allocate_work_area);
  LOG_ERR_IF(u_hat.memory_location() != zisa::device_type::cuda,
             "Unsupported Memory Location");
  LOG_ERR_IF(u.memory_location() != zisa::device_type::cuda,
             "Unsupported Memory Location");

  zisa::shape_t<3> partial_u_hat_size(
      u.shape(0), u.shape(1), u.shape(2) / 2 + 1);
  partial_u_hat_ = zisa::cuda_array<complex_t, 3>(partial_u_hat_size);
  mpi_send_buffer_ = zisa::array<complex_t, 3>(partial_u_hat_size);
  mpi_recv_buffer_ = zisa::array<complex_t, 3>(u_hat_.shape());

  size_t workspace_size = 0;
  // Create the plans for the forward operations
  if (direction_ && FFT_FORWARD) {
    // Before transposing
    cufftResult status = cufftCreate(&plan_forward_r2c_);
    cudaCheckError(status);
    status = cufftSetAutoAllocation(plan_forward_r2c_, false);
    cudaCheckError(status);
    size_t fwd1_size;
    status = cufftMakePlan1d(plan_forward_r2c_,         // plan
                             u_.shape(2),               // n
                             type_forward_r2c,          // type
                             u_.shape(0) * u_.shape(1), // batch
                             &fwd1_size);               // workSize
    cudaCheckError(status);
    workspace_size = std::max(workspace_size, fwd1_size);
    // After transposing
    status = cufftCreate(&plan_forward_c2c_);
    cudaCheckError(status);
    status = cufftSetAutoAllocation(plan_forward_c2c_, false);
    cudaCheckError(status);
    size_t fwd2_size;
    status = cufftMakePlan1d(plan_forward_c2c_,                 // plan
                             u_hat_.shape(2),                   // n
                             type_forward_c2c,                  // type
                             u_hat_.shape(0) * u_hat_.shape(1), // batch
                             &fwd2_size);                       // workSize
    cudaCheckError(status);
    workspace_size = std::max(workspace_size, fwd2_size);
  }
  // Create the plans for the backward operations
  if (direction_ & FFT_BACKWARD) {
    // Before transposing
    cufftResult status = cufftCreate(&plan_backward_c2c_);
    cudaCheckError(status);
    status = cufftSetAutoAllocation(plan_backward_c2c_, false);
    cudaCheckError(status);
    size_t bkw1_size;
    status = cufftMakePlan1d(plan_backward_c2c_,                // plan
                             u_hat_.shape(2),                   // n
                             type_backward_c2c,                 // type
                             u_hat_.shape(0) * u_hat_.shape(1), // batch
                             &bkw1_size);                       // workSize
    cudaCheckError(status);
    workspace_size = std::max(workspace_size, bkw1_size);
    // After transposing
    status = cufftCreate(&plan_backward_c2r_);
    cudaCheckError(status);
    status = cufftSetAutoAllocation(plan_backward_c2r_, false);
    cudaCheckError(status);
    size_t bkw2_size;
    status = cufftMakePlan1d(plan_backward_c2r_,        // plan
                             u_.shape(2),               // n
                             type_backward_c2r,         // type
                             u_.shape(0) * u_.shape(1), // batch
                             &bkw2_size);               // workSize
    cudaCheckError(status);
    workspace_size = std::max(workspace_size, bkw2_size);
  }
  // Allocate the shared work area
  if (work_area_ == nullptr) {
    cudaError_t status = cudaMalloc((void **)(&work_area_), workspace_size);
    cudaCheckError(status);
  }
  if (direction_ & FFT_FORWARD) {
    cufftResult status = cufftSetWorkArea(plan_forward_r2c_, work_area_);
    cudaCheckError(status);
    status = cufftSetWorkArea(plan_forward_c2c_, work_area_);
    cudaCheckError(status);
  }
  if (direction_ & FFT_BACKWARD) {
    cufftResult status = cufftSetWorkArea(plan_backward_c2r_, work_area_);
    cudaCheckError(status);
    status = cufftSetWorkArea(plan_backward_c2c_, work_area_);
    cudaCheckError(status);
  }
}

void CUFFT_MPI<2>::do_set_work_area(void *work_area) { ZISA_UNUSED(work_area); }

CUFFT_MPI<3>::CUFFT_MPI(const zisa::array_view<complex_t, 4> &u_hat,
                        const zisa::array_view<real_t, 4> &u,
                        const Communicator *comm,
                        int direction,
                        void *work_area)
    : super(direction),
      comm_(comm),
      work_area_(work_area),
      free_work_area_(work_area == nullptr) {
  initialize(u_hat, u);
}

CUFFT_MPI<3>::~CUFFT_MPI() {
  if (direction_ & FFT_FORWARD) {
    cufftDestroy(plan_forward_r2c_);
    cufftDestroy(plan_forward_c2c_);
  }
  if (direction_ & FFT_BACKWARD) {
    cufftDestroy(plan_backward_c2r_);
    cufftDestroy(plan_backward_c2c_);
  }
  if (free_work_area_) {
    cudaFree(work_area_);
  }
}

size_t CUFFT_MPI<3>::get_work_area_size() const { return 0; }

void CUFFT_MPI<3>::forward() {
  LOG_ERR_IF((direction_ & FFT_FORWARD) == 0,
             "Forward operation was not initialized");
  AZEBAN_PROFILE_START("CUFFT_MPI::forward", comm_->get_mpi_comm());
  // TODO: Remove the C-style casts when the compiler chooses not to ignore the
  // `constexpr` anymore
  if constexpr (std::is_same_v<float, real_t>) {
    // Perform the local FFTs
    cufftResult status
        = cufftExecR2C(plan_forward_r2c_,
                       (float *)u_.raw(),
                       reinterpret_cast<cufftComplex *>(partial_u_hat_.raw()));
    cudaCheckError(status);
    cudaDeviceSynchronize();
    // Transpose the data from partial_u_hat_ to u_hat_
    AZEBAN_PROFILE_START("CUFFT_MPI::transpose", comm_->get_mpi_comm());
    zisa::copy(mpi_send_buffer_, partial_u_hat_);
    transpose(mpi_recv_buffer_, mpi_send_buffer_, comm_);
    zisa::copy(u_hat_, mpi_recv_buffer_);
    AZEBAN_PROFILE_STOP("CUFFT_MPI::transpose", comm_->get_mpi_comm());
    // Perform the final local FFTs in place
    status = cufftExecC2C(plan_forward_c2c_,
                          reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                          reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                          CUFFT_FORWARD);
    cudaCheckError(status);
    cudaDeviceSynchronize();
  } else {
    // Perform the local FFTs
    cufftResult status = cufftExecD2Z(
        plan_forward_r2c_,
        (double *)u_.raw(),
        reinterpret_cast<cufftDoubleComplex *>(partial_u_hat_.raw()));
    cudaCheckError(status);
    cudaDeviceSynchronize();
    // Transpose the data from partial_u_hat_ to u_hat_
    AZEBAN_PROFILE_START("CUFFT_MPI::transpose");
    zisa::copy(mpi_send_buffer_, partial_u_hat_);
    transpose(mpi_recv_buffer_, mpi_send_buffer_, comm_);
    zisa::copy(u_hat_, mpi_recv_buffer_);
    AZEBAN_PROFILE_STOP("CUFFT_MPI::transpose");
    // Perform the final local FFTs in place
    status = cufftExecZ2Z(plan_forward_c2c_,
                          reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                          reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                          CUFFT_FORWARD);
    cudaCheckError(status);
    cudaDeviceSynchronize();
  }
  AZEBAN_PROFILE_STOP("CUFFT_MPI::forward", comm_->get_mpi_comm());
}

void CUFFT_MPI<3>::backward() {
  LOG_ERR_IF((direction_ & FFT_BACKWARD) == 0,
             "Backward operation was not initialized");
  AZEBAN_PROFILE_START("CUFFT_MPI::backward", comm_->get_mpi_comm());
  // TODO: Remove the C-style casts when the compiler chooses not to ignore the
  // `constexpr` anymore
  if constexpr (std::is_same_v<float, real_t>) {
    // Perform the local FFTs
    cufftResult status
        = cufftExecC2C(plan_backward_c2c_,
                       reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                       reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                       CUFFT_INVERSE);
    cudaCheckError(status);
    cudaDeviceSynchronize();
    // Transpose the data from partial_u_hat_ to u_hat_
    AZEBAN_PROFILE_START("CUFFT_MPI::transpose", comm_->get_mpi_comm());
    zisa::copy(mpi_recv_buffer_, u_hat_);
    transpose(mpi_send_buffer_, mpi_recv_buffer_, comm_);
    zisa::copy(partial_u_hat_, mpi_send_buffer_);
    AZEBAN_PROFILE_STOP("CUFFT_MPI::transpose", comm_->get_mpi_comm());
    // Perform the final local FFTs in place
    status
        = cufftExecC2R(plan_backward_c2r_,
                       reinterpret_cast<cufftComplex *>(partial_u_hat_.raw()),
                       (float *)u_.raw());
    cudaCheckError(status);
    cudaDeviceSynchronize();
  } else {
    // Perform the local FFTs
    cufftResult status
        = cufftExecZ2Z(plan_backward_c2c_,
                       reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                       reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                       CUFFT_INVERSE);
    cudaCheckError(status);
    cudaDeviceSynchronize();
    // Transpose the data from partial_u_hat_ to u_hat_
    AZEBAN_PROFILE_START("CUFFT_MPI::transpose", comm_->get_mpi_comm());
    zisa::copy(mpi_recv_buffer_, u_hat_);
    transpose(mpi_send_buffer_, mpi_recv_buffer_, comm_);
    zisa::copy(partial_u_hat_, mpi_send_buffer_);
    AZEBAN_PROFILE_STOP("CUFFT_MPI::transpose", comm_->get_mpi_comm());
    // Perform the final local FFTs in place
    status = cufftExecZ2D(
        plan_backward_c2r_,
        reinterpret_cast<cufftDoubleComplex *>(partial_u_hat_.raw()),
        (double *)u_.raw());
    cudaCheckError(status);
    cudaDeviceSynchronize();
  }
  AZEBAN_PROFILE_STOP("CUFFT_MPI::backward", comm_->get_mpi_comm());
}

void *CUFFT_MPI<3>::get_work_area() const { return work_area_; }

void CUFFT_MPI<3>::do_initialize(const zisa::array_view<complex_t, 4> &u_hat,
                                 const zisa::array_view<real_t, 4> &u,
                                 bool allocate_work_area) {
  ZISA_UNUSED(allocate_work_area);
  LOG_ERR_IF(u_hat.memory_location() != zisa::device_type::cuda,
             "Unsupported Memory Location");
  LOG_ERR_IF(u.memory_location() != zisa::device_type::cuda,
             "Unsupported Memory Location");

  zisa::shape_t<4> partial_u_hat_size(
      u.shape(0), u.shape(1), u.shape(2), u.shape(3) / 2 + 1);
  partial_u_hat_ = zisa::cuda_array<complex_t, 4>(partial_u_hat_size);
  mpi_send_buffer_ = zisa::array<complex_t, 4>(partial_u_hat_size);
  mpi_recv_buffer_ = zisa::array<complex_t, 4>(u_hat_.shape());

  size_t workspace_size = 0;
  // Create the plans for the forward operations
  if (direction_ && FFT_FORWARD) {
    // Before transposing
    cufftResult status = cufftCreate(&plan_forward_r2c_);
    cudaCheckError(status);
    status = cufftSetAutoAllocation(plan_forward_r2c_, false);
    cudaCheckError(status);
    int n[2] = {zisa::integer_cast<int>(u_.shape(2)),
                zisa::integer_cast<int>(u_.shape(3))};
    size_t fwd1_size;
    status = cufftMakePlanMany(plan_forward_r2c_,                 // plan
                               2,                                 // rank
                               n,                                 // n
                               NULL,                              // inembed
                               1,                                 // istride
                               u_.shape(2) * u_.shape(3),         // idist
                               NULL,                              // onembed
                               1,                                 // ostride
                               u_hat_.shape(2) * u_hat_.shape(3), // odist
                               type_forward_r2c,                  // type
                               u_.shape(0) * u_.shape(1),         // batch
                               &fwd1_size);                       // workSize
    cudaCheckError(status);
    workspace_size = std::max(workspace_size, fwd1_size);
    // After transposing
    status = cufftCreate(&plan_forward_c2c_);
    cudaCheckError(status);
    status = cufftSetAutoAllocation(plan_forward_c2c_, false);
    cudaCheckError(status);
    size_t fwd2_size;
    status = cufftMakePlan1d(plan_forward_c2c_, // plan
                             u_hat_.shape(3),   // n
                             type_forward_c2c,  // type
                             u_hat_.shape(0) * u_hat_.shape(1)
                                 * u_hat_.shape(2), // batch
                             &fwd2_size);           // workSize
    cudaCheckError(status);
    workspace_size = std::max(workspace_size, fwd2_size);
  }
  // Create the plans for the backward operations
  if (direction_ & FFT_BACKWARD) {
    // Before transposing
    cufftResult status = cufftCreate(&plan_backward_c2c_);
    cudaCheckError(status);
    status = cufftSetAutoAllocation(plan_backward_c2c_, false);
    cudaCheckError(status);
    size_t bkw1_size;
    status = cufftMakePlan1d(plan_backward_c2c_, // plan
                             u_hat_.shape(3),    // n
                             type_backward_c2c,  // type
                             u_hat_.shape(0) * u_hat_.shape(1)
                                 * u_hat_.shape(2), // batch
                             &bkw1_size);           // workSize
    cudaCheckError(status);
    workspace_size = std::max(workspace_size, bkw1_size);
    // After transposing
    status = cufftCreate(&plan_backward_c2r_);
    cudaCheckError(status);
    status = cufftSetAutoAllocation(plan_backward_c2r_, false);
    cudaCheckError(status);
    int n[2] = {zisa::integer_cast<int>(u_.shape(2)),
                zisa::integer_cast<int>(u_.shape(3))};
    size_t bkw2_size;
    status = cufftMakePlanMany(plan_backward_c2r_, // plan
                               2,                  // rank
                               n,                  // n
                               NULL,               // inembed
                               1,                  // istride
                               partial_u_hat_.shape(2)
                                   * partial_u_hat_.shape(3), // idist
                               NULL,                          // onembed
                               1,                             // ostride
                               u_.shape(2) * u_.shape(3),     // odist,
                               type_backward_c2r,             // type,
                               u_.shape(0) * u_.shape(1),     // batch
                               &bkw2_size);                   // workSize
    cudaCheckError(status);
    workspace_size = std::max(workspace_size, bkw2_size);
  }
  // Allocate the shared work area
  if (work_area_ == nullptr) {
    cudaError_t status = cudaMalloc((void **)(&work_area_), workspace_size);
    cudaCheckError(status);
  }
  if (direction_ & FFT_FORWARD) {
    cufftResult status = cufftSetWorkArea(plan_forward_r2c_, work_area_);
    cudaCheckError(status);
    status = cufftSetWorkArea(plan_forward_c2c_, work_area_);
    cudaCheckError(status);
  }
  if (direction_ & FFT_BACKWARD) {
    cufftResult status = cufftSetWorkArea(plan_backward_c2r_, work_area_);
    cudaCheckError(status);
    status = cufftSetWorkArea(plan_backward_c2c_, work_area_);
    cudaCheckError(status);
  }
}

void CUFFT_MPI<3>::do_set_work_area(void *work_area) { ZISA_UNUSED(work_area); }

}
#endif
