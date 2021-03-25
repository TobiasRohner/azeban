#include <azeban/cuda/operations/cufft_mpi.hpp>
#include <azeban/mpi_types.hpp>
#include <azeban/profiler.hpp>
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
  MPI_Type_vector(N_phys, 1, u_.shape(2), mpi_type<complex_t>(), &col_type_);
  MPI_Type_commit(&col_type_);
  natural_types_ = std::vector<MPI_Datatype>(size);
  transposed_types_ = std::vector<MPI_Datatype>(size);
  for (int r = 0; r < size; ++r) {
    MPI_Type_create_hvector(
        size_u_hat_[r], 1, sizeof(complex_t), col_type_, &natural_types_[r]);
    MPI_Type_commit(&natural_types_[r]);
    MPI_Type_vector(N_fourier,
                    size_u_[r],
                    u_hat_.shape(2),
                    mpi_type<complex_t>(),
                    &transposed_types_[r]);
    MPI_Type_commit(&transposed_types_[r]);
  }

  zisa::shape_t<3> partial_u_hat_size(
      u.shape(0), u.shape(1), u.shape(2) / 2 + 1);
  partial_u_hat_ = zisa::cuda_array<complex_t, 3>(partial_u_hat_size);

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
    // TODO: Implement
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
    // TODO: Implement
  }
}

CUFFT_MPI<2>::~CUFFT_MPI() {
  if (direction_ & FFT_FORWARD) {
    cufftDestroy(plan_forward_r2c_);
    cufftDestroy(plan_forward_c2c_);
  }
  if (direction_ & FFT_BACKWARD) {
    // TODO: Implement
  }
  cudaFree(work_area_);
  MPI_Type_free(&col_type_);
  for (auto &&t : natural_types_) {
    MPI_Type_free(&t);
  }
  for (auto &&t : transposed_types_) {
    MPI_Type_free(&t);
  }
}

void CUFFT_MPI<2>::forward() {
  LOG_ERR_IF((direction_ & FFT_FORWARD) == 0,
             "Forward operation was not initialized");
  AZEBAN_PROFILE_START("CUFFT_MPI::forward");
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
  AZEBAN_PROFILE_STOP("CUFFT_MPI::forward");
}

void CUFFT_MPI<2>::backward() {
  // TODO
}

void CUFFT_MPI<2>::transpose_forward() {
  int rank, size;
  MPI_Comm_rank(comm_, &rank);
  MPI_Comm_size(comm_, &size);

  const void *sendbuf = (void *)partial_u_hat_.raw();
  const std::vector<int> sendcounts(size, 1);
  std::vector<int> sdispls(size);
  const MPI_Datatype *sendtypes = natural_types_.data();
  void *recvbuf = (void *)u_hat_.raw();
  const std::vector<int> recvcounts(size, 1);
  std::vector<int> rdispls(size);
  const MPI_Datatype *recvtypes = transposed_types_.data();
  sdispls[0] = 0;
  rdispls[0] = 0;
  for (int r = 1; r < size; ++r) {
    sdispls[r] = sdispls[r - 1] + size_u_hat_[r - 1] * sizeof(complex_t);
    rdispls[r] = rdispls[r - 1] + size_u_[r - 1] * sizeof(complex_t);
  }
  MPI_Alltoallw(sendbuf,
                sendcounts.data(),
                sdispls.data(),
                sendtypes,
                recvbuf,
                recvcounts.data(),
                rdispls.data(),
                recvtypes,
                comm_);
}

void CUFFT_MPI<2>::transpose_backward() {
  // TODO: Implement
}

}
