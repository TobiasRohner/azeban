#if AZEBAN_HAS_MPI
#include <azeban/cuda/operations/cufft_mpi.hpp>
#include <azeban/mpi_types.hpp>
#include <azeban/profiler.hpp>
#include <azeban/operations/transpose.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>

namespace azeban {

CUFFT_MPI<2>::CUFFT_MPI(const zisa::array_view<complex_t, 3> &u_hat,
                        const zisa::array_view<real_t, 3> &u,
                        MPI_Comm comm,
                        int direction)
    : super(u_hat, u, direction), comm_(comm) {
  LOG_ERR_IF(u_hat.memory_location() != zisa::device_type::cuda,
             "Unsupported Memory Location");
  LOG_ERR_IF(u.memory_location() != zisa::device_type::cuda,
             "Unsupported Memory Location");

  int rank, size;
  MPI_Comm_rank(comm_, &rank);
  MPI_Comm_size(comm_, &size);

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
  cudaMalloc((void **)(&work_area_), workspace_size);
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

CUFFT_MPI<2>::~CUFFT_MPI() {
  if (direction_ & FFT_FORWARD) {
    cufftDestroy(plan_forward_r2c_);
    cufftDestroy(plan_forward_c2c_);
  }
  if (direction_ & FFT_BACKWARD) {
    cufftDestroy(plan_backward_c2r_);
    cufftDestroy(plan_backward_c2c_);
  }
  cudaFree(work_area_);
}

void CUFFT_MPI<2>::forward() {
  LOG_ERR_IF((direction_ & FFT_FORWARD) == 0,
             "Forward operation was not initialized");
  AZEBAN_PROFILE_START("CUFFT_MPI::forward", comm_);
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
    zisa::copy(mpi_send_buffer_, partial_u_hat_);
    transpose(mpi_recv_buffer_, mpi_send_buffer_, comm_);
    zisa::copy(u_hat_, mpi_recv_buffer_);
    // Perform the final local FFTs in place
    status = cufftExecC2C(plan_forward_c2c_,
                          reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                          reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                          CUFFT_FORWARD);
    cudaCheckError(status);
  } else {
    // Perform the local FFTs
    cufftResult status = cufftExecD2Z(
        plan_forward_r2c_,
        (double *)u_.raw(),
        reinterpret_cast<cufftDoubleComplex *>(partial_u_hat_.raw()));
    cudaCheckError(status);
    cudaDeviceSynchronize();
    // Transpose the data from partial_u_hat_ to u_hat_
    zisa::copy(mpi_send_buffer_, partial_u_hat_);
    transpose(mpi_recv_buffer_, mpi_send_buffer_, comm_);
    zisa::copy(u_hat_, mpi_recv_buffer_);
    // Perform the final local FFTs in place
    status = cufftExecZ2Z(plan_forward_c2c_,
                          reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                          reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                          CUFFT_FORWARD);
    cudaCheckError(status);
  }
  AZEBAN_PROFILE_STOP("CUFFT_MPI::forward", comm_);
}

void CUFFT_MPI<2>::backward() {
  LOG_ERR_IF((direction_ & FFT_BACKWARD) == 0,
             "Backward operation was not initialized");
  AZEBAN_PROFILE_START("CUFFT_MPI::backward", comm_);
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
    zisa::copy(mpi_recv_buffer_, u_hat_);
    transpose(mpi_send_buffer_, mpi_recv_buffer_, comm_);
    zisa::copy(partial_u_hat_, mpi_send_buffer_);
    // Perform the final local FFTs in place
    status
        = cufftExecC2R(plan_forward_c2c_,
                       reinterpret_cast<cufftComplex *>(partial_u_hat_.raw()),
                       (float *)u_.raw());
    cudaCheckError(status);
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
    zisa::copy(mpi_recv_buffer_, u_hat_);
    transpose(mpi_send_buffer_, mpi_recv_buffer_, comm_);
    zisa::copy(partial_u_hat_, mpi_send_buffer_);
    // Perform the final local FFTs in place
    status = cufftExecZ2D(
        plan_backward_c2r_,
        reinterpret_cast<cufftDoubleComplex *>(partial_u_hat_.raw()),
        (double *)u_.raw());
    cudaCheckError(status);
  }
  AZEBAN_PROFILE_STOP("CUFFT_MPI::backward", comm_);
}

CUFFT_MPI<3>::CUFFT_MPI(const zisa::array_view<complex_t, 4> &u_hat,
                        const zisa::array_view<real_t, 4> &u,
                        MPI_Comm comm,
                        int direction)
    : super(u_hat, u, direction), comm_(comm) {
  LOG_ERR_IF(u_hat.memory_location() != zisa::device_type::cuda,
             "Unsupported Memory Location");
  LOG_ERR_IF(u.memory_location() != zisa::device_type::cuda,
             "Unsupported Memory Location");

  int rank, size;
  MPI_Comm_rank(comm_, &rank);
  MPI_Comm_size(comm_, &size);

  zisa::shape_t<4> partial_u_hat_size(
      u.shape(0), u.shape(1), u.shape(2), u.shape(3) / 2 + 1);
  partial_u_hat_ = zisa::cuda_array<complex_t, 4>(partial_u_hat_size);
  mpi_send_buffer_ = zisa::array<complex_t, 4>(partial_u_hat_size);
  mpi_recv_buffer_ = zisa::array<complex_t, 4>(u_hat_.shape());

  // Find out how big the data chunks of each rank are
  const zisa::int_t N_phys = u_.shape(1);
  const zisa::int_t N_fourier = u_hat_.shape(1);
  size_u_ = std::make_unique<zisa::int_t[]>(size);
  size_u_hat_ = std::make_unique<zisa::int_t[]>(size);
  MPI_Allgather(
      &N_phys, 1, mpi_type(N_phys), size_u_.get(), 1, mpi_type(N_phys), comm_);
  MPI_Allgather(&N_fourier,
                1,
                mpi_type(N_fourier),
                size_u_hat_.get(),
                1,
                mpi_type(N_fourier),
                comm_);

  // Construct datatypes to transmit and receive rows and columns of data
  MPI_Datatype col_type_large;
  MPI_Datatype col_type;
  MPI_Type_vector(N_phys,
                  1,
                  partial_u_hat_.shape(2) * partial_u_hat_.shape(3),
                  mpi_type<complex_t>(),
                  &col_type_large);
  MPI_Type_commit(&col_type_large);
  MPI_Type_create_resized(col_type_large, 0, sizeof(complex_t), &col_type);
  MPI_Type_commit(&col_type);
  MPI_Type_free(&col_type_large);
  natural_types_ = std::vector<MPI_Datatype>(size);
  transposed_types_ = std::vector<MPI_Datatype>(size);
  for (int r = 0; r < size; ++r) {
    int sizes_nat[2] = {zisa::integer_cast<int>(partial_u_hat_.shape(2)),
                        zisa::integer_cast<int>(partial_u_hat_.shape(3))};
    int subsizes_nat[2] = {zisa::integer_cast<int>(size_u_hat_[r]),
                           zisa::integer_cast<int>(partial_u_hat_.shape(3))};
    int starts_nat[2] = {0, 0};
    MPI_Type_create_subarray(2,
                             sizes_nat,
                             subsizes_nat,
                             starts_nat,
                             MPI_ORDER_C,
                             col_type,
                             &natural_types_[r]);
    MPI_Type_commit(&natural_types_[r]);
    int sizes_trans[3] = {zisa::integer_cast<int>(u_hat_.shape(1)),
                          zisa::integer_cast<int>(u_hat_.shape(2)),
                          zisa::integer_cast<int>(u_hat_.shape(3))};
    int subsizes_trans[3] = {zisa::integer_cast<int>(u_hat_.shape(1)),
                             zisa::integer_cast<int>(u_hat_.shape(2)),
                             zisa::integer_cast<int>(size_u_[r])};
    int starts_trans[3] = {0, 0, 0};
    MPI_Type_create_subarray(3,
                             sizes_trans,
                             subsizes_trans,
                             starts_trans,
                             MPI_ORDER_C,
                             mpi_type<complex_t>(),
                             &transposed_types_[r]);
    MPI_Type_commit(&transposed_types_[r]);
  }
  MPI_Type_free(&col_type);

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
  cudaMalloc((void **)(&work_area_), workspace_size);
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

CUFFT_MPI<3>::~CUFFT_MPI() {
  if (direction_ & FFT_FORWARD) {
    cufftDestroy(plan_forward_r2c_);
    cufftDestroy(plan_forward_c2c_);
  }
  if (direction_ & FFT_BACKWARD) {
    cufftDestroy(plan_backward_c2r_);
    cufftDestroy(plan_backward_c2c_);
  }
  cudaFree(work_area_);
  for (auto &&t : natural_types_) {
    MPI_Type_free(&t);
  }
  for (auto &&t : transposed_types_) {
    MPI_Type_free(&t);
  }
}

void CUFFT_MPI<3>::forward() {
  LOG_ERR_IF((direction_ & FFT_FORWARD) == 0,
             "Forward operation was not initialized");
  int rank;
  MPI_Comm_rank(comm_, &rank);
  AZEBAN_PROFILE_START("CUFFT_MPI::forward", comm_);
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
    transpose_forward();
    // Perform the final local FFTs in place
    status = cufftExecC2C(plan_forward_c2c_,
                          reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                          reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                          CUFFT_FORWARD);
    cudaCheckError(status);
  } else {
    // Perform the local FFTs
    cufftResult status = cufftExecD2Z(
        plan_forward_r2c_,
        (double *)u_.raw(),
        reinterpret_cast<cufftDoubleComplex *>(partial_u_hat_.raw()));
    cudaCheckError(status);
    cudaDeviceSynchronize();
    // Transpose the data from partial_u_hat_ to u_hat_
    transpose_forward();
    // Perform the final local FFTs in place
    status = cufftExecZ2Z(plan_forward_c2c_,
                          reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                          reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                          CUFFT_FORWARD);
    cudaCheckError(status);
  }
  AZEBAN_PROFILE_STOP("CUFFT_MPI::forward", comm_);
}

void CUFFT_MPI<3>::backward() {
  LOG_ERR_IF((direction_ & FFT_BACKWARD) == 0,
             "Backward operation was not initialized");
  int rank;
  MPI_Comm_rank(comm_, &rank);
  AZEBAN_PROFILE_START("CUFFT_MPI::backward", comm_);
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
    transpose_backward();
    // Perform the final local FFTs in place
    status
        = cufftExecC2R(plan_forward_c2c_,
                       reinterpret_cast<cufftComplex *>(partial_u_hat_.raw()),
                       (float *)u_.raw());
    cudaCheckError(status);
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
    transpose_backward();
    // Perform the final local FFTs in place
    status = cufftExecZ2D(
        plan_backward_c2r_,
        reinterpret_cast<cufftDoubleComplex *>(partial_u_hat_.raw()),
        (double *)u_.raw());
    cudaCheckError(status);
  }
  AZEBAN_PROFILE_STOP("CUFFT_MPI::backward", comm_);
}

void CUFFT_MPI<3>::transpose_forward() {
  AZEBAN_PROFILE_START("CUFFT_MPI::transpose_forward", comm_);
  int rank, size;
  MPI_Comm_rank(comm_, &rank);
  MPI_Comm_size(comm_, &size);

  const std::vector<int> sendcounts(size, 1);
  const std::vector<int> recvcounts(size, 1);
  std::vector<int> sdispls(size);
  std::vector<int> rdispls(size);
  sdispls[0] = 0;
  rdispls[0] = 0;
  for (int r = 1; r < size; ++r) {
    sdispls[r]
        = sdispls[r - 1]
          + size_u_hat_[r - 1] * partial_u_hat_.shape(3) * sizeof(complex_t);
    rdispls[r] = rdispls[r - 1] + size_u_[r - 1] * sizeof(complex_t);
  }

  zisa::copy(mpi_send_buffer_, partial_u_hat_);

  const zisa::int_t N = mpi_send_buffer_.shape(0);
  std::vector<MPI_Request> reqs(N);
  for (zisa::int_t d = 0; d < N; ++d) {
    const ptrdiff_t send_offset
        = d * zisa::product(mpi_send_buffer_.shape()) / N;
    const ptrdiff_t recv_offset
        = d * zisa::product(mpi_recv_buffer_.shape()) / N;
    MPI_Ialltoallw(mpi_send_buffer_.raw() + send_offset,
                   sendcounts.data(),
                   sdispls.data(),
                   natural_types_.data(),
                   mpi_recv_buffer_.raw() + recv_offset,
                   recvcounts.data(),
                   rdispls.data(),
                   transposed_types_.data(),
                   comm_,
                   &reqs[d]);
  }
  MPI_Waitall(N, reqs.data(), MPI_STATUSES_IGNORE);

  zisa::copy(u_hat_, mpi_recv_buffer_);
  AZEBAN_PROFILE_STOP("CUFFT_MPI::transpose_forward", comm_);
}

void CUFFT_MPI<3>::transpose_backward() {
  AZEBAN_PROFILE_START("CUFFT_MPI::transpose_backward", comm_);
  int rank, size;
  MPI_Comm_rank(comm_, &rank);
  MPI_Comm_size(comm_, &size);

  const std::vector<int> sendcounts(size, 1);
  const std::vector<int> recvcounts(size, 1);
  std::vector<int> sdispls(size);
  std::vector<int> rdispls(size);
  sdispls[0] = 0;
  rdispls[0] = 0;
  for (int r = 1; r < size; ++r) {
    sdispls[r] = sdispls[r - 1] + size_u_[r - 1] * sizeof(complex_t);
    rdispls[r]
        = rdispls[r - 1]
          + size_u_hat_[r - 1] * partial_u_hat_.shape(3) * sizeof(complex_t);
  }

  zisa::copy(mpi_recv_buffer_, u_hat_);

  const zisa::int_t N = mpi_send_buffer_.shape(0);
  std::vector<MPI_Request> reqs(N);
  for (zisa::int_t d = 0; d < N; ++d) {
    const ptrdiff_t send_offset
        = d * zisa::product(mpi_recv_buffer_.shape()) / N;
    const ptrdiff_t recv_offset
        = d * zisa::product(mpi_send_buffer_.shape()) / N;
    MPI_Ialltoallw(mpi_recv_buffer_.raw() + send_offset,
                   sendcounts.data(),
                   sdispls.data(),
                   transposed_types_.data(),
                   mpi_send_buffer_.raw() + recv_offset,
                   recvcounts.data(),
                   rdispls.data(),
                   natural_types_.data(),
                   comm_,
                   &reqs[d]);
  }
  MPI_Waitall(N, reqs.data(), MPI_STATUSES_IGNORE);

  zisa::copy(partial_u_hat_, mpi_send_buffer_);
  AZEBAN_PROFILE_STOP("CUFFT_MPI::transpose_backward", comm_);
}

}
#endif
