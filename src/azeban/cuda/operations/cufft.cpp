#include <azeban/cuda/cuda_check_error.hpp>
#include <azeban/cuda/operations/cufft.hpp>
#include <azeban/profiler.hpp>
#include <zisa/utils/logging.hpp>

namespace azeban {

static cufftResult execute_c2c(cufftHandle plan,
                               Complex<float> *idata,
                               Complex<float> *odata,
                               fft_direction direction) {
  int sign;
  if (direction == FFT_FORWARD) {
    sign = CUFFT_FORWARD;
  } else {
    sign = CUFFT_INVERSE;
  }
  return cufftExecC2C(plan,
                      reinterpret_cast<cufftComplex *>(idata),
                      reinterpret_cast<cufftComplex *>(odata),
                      sign);
}

static cufftResult execute_c2c(cufftHandle plan,
                               Complex<double> *idata,
                               Complex<double> *odata,
                               fft_direction direction) {
  int sign;
  if (direction == FFT_FORWARD) {
    sign = CUFFT_FORWARD;
  } else {
    sign = CUFFT_INVERSE;
  }
  return cufftExecZ2Z(plan,
                      reinterpret_cast<cufftDoubleComplex *>(idata),
                      reinterpret_cast<cufftDoubleComplex *>(odata),
                      sign);
}

static cufftResult
execute_r2c(cufftHandle plan, float *idata, Complex<float> *odata) {
  return cufftExecR2C(plan, idata, reinterpret_cast<cufftComplex *>(odata));
}

static cufftResult
execute_r2c(cufftHandle plan, double *idata, Complex<double> *odata) {
  return cufftExecD2Z(
      plan, idata, reinterpret_cast<cufftDoubleComplex *>(odata));
}

static cufftResult
execute_c2r(cufftHandle plan, Complex<float> *idata, float *odata) {
  return cufftExecC2R(plan, reinterpret_cast<cufftComplex *>(idata), odata);
}

static cufftResult
execute_c2r(cufftHandle plan, Complex<double> *idata, double *odata) {
  return cufftExecZ2D(
      plan, reinterpret_cast<cufftDoubleComplex *>(idata), odata);
}

template <int Dim>
CUFFT_R2C<Dim>::CUFFT_R2C(const zisa::array_view<complex_t, dim_v + 1> &u_hat,
                          const zisa::array_view<real_t, dim_v + 1> &u,
                          int direction)
    : super(direction) {
  initialize(u_hat, u);
}

template <int Dim>
CUFFT_R2C<Dim>::~CUFFT_R2C() {
  if (is_forward()) {
    cufftDestroy(plan_forward_);
  }
  if (is_backward()) {
    cufftDestroy(plan_backward_);
  }
  cudaFree(work_area_);
}

template <int Dim>
void CUFFT_R2C<Dim>::forward() {
  LOG_ERR_IF(!is_forward(), "Forward operation was not initialized");
  AZEBAN_PROFILE_START("CUFFT_R2C::forward");
  auto status = execute_r2c(plan_forward_, u_.raw(), u_hat_.raw());
  cudaCheckError(status);
  cudaDeviceSynchronize();
  AZEBAN_PROFILE_STOP("CUFFT_R2C::forward");
}

template <int Dim>
void CUFFT_R2C<Dim>::backward() {
  LOG_ERR_IF(!is_backward(), "Backward operation was not initialized");
  AZEBAN_PROFILE_START("CUFFT_R2C::backward");
  auto status = execute_c2r(plan_backward_, u_hat_.raw(), u_.raw());
  cudaCheckError(status);
  cudaDeviceSynchronize();
  AZEBAN_PROFILE_STOP("CUFFT_R2C::backward");
}

template <int Dim>
void CUFFT_R2C<Dim>::do_initialize(
    const zisa::array_view<complex_t, Dim + 1> &u_hat,
    const zisa::array_view<real_t, Dim + 1> &u) {
  LOG_ERR_IF(u_hat.memory_location() != zisa::device_type::cuda,
             "cuFFT is GPU only!");
  LOG_ERR_IF(u.memory_location() != zisa::device_type::cuda,
             "cuFFT is GPU only!");
  int rdist = 1;
  int cdist = 1;
  int n[dim_v];
  for (int i = 0; i < dim_v; ++i) {
    rdist *= u.shape(i + 1);
    cdist *= u_hat.shape(i + 1);
    n[i] = u.shape(i + 1);
  }
  size_t workspace_size = 0;
  // Create a plan for the forward operation
  if (is_forward()) {
    cufftResult status = cufftCreate(&plan_forward_);
    cudaCheckError(status);
    status = cufftSetAutoAllocation(plan_forward_, false);
    cudaCheckError(status);
    status = cufftPlanMany(&plan_forward_, // plan
                           dim_v,          // rank
                           n,              // n
                           NULL,           // inembed
                           1,              // istride
                           rdist,          // idist
                           NULL,           // onembed
                           1,              // ostride
                           cdist,          // odist
                           type_forward,   // type
                           u.shape(0));    // batch
    cudaCheckError(status);
    size_t fwd_size;
    status = cufftGetSize(plan_forward_, &fwd_size);
    cudaCheckError(status);
    workspace_size = std::max(workspace_size, fwd_size);
  }
  // Create a plan for the backward operation
  if (is_backward()) {
    cufftResult status = cufftCreate(&plan_backward_);
    cudaCheckError(status);
    status = cufftSetAutoAllocation(plan_backward_, false);
    cudaCheckError(status);
    status = cufftPlanMany(&plan_backward_, // plan
                           dim_v,           // rank
                           n,               // n
                           NULL,            // inembed
                           1,               // istride
                           cdist,           // idist
                           NULL,            // onembed
                           1,               // ostride
                           rdist,           // odist
                           type_backward,   // type
                           u.shape(0));     // batch
    cudaCheckError(status);
    size_t bkw_size;
    status = cufftGetSize(plan_backward_, &bkw_size);
    cudaCheckError(status);
    workspace_size = std::max(workspace_size, bkw_size);
  }
  // Allocate the shared work area
  cudaMalloc((void **)(&work_area_), workspace_size);
  if (direction_ & FFT_FORWARD) {
    cufftResult status = cufftSetWorkArea(plan_forward_, work_area_);
    cudaCheckError(status);
  }
  if (direction_ & FFT_BACKWARD) {
    cufftResult status = cufftSetWorkArea(plan_backward_, work_area_);
    cudaCheckError(status);
  }
}

template <int Dim>
CUFFT_C2C<Dim>::CUFFT_C2C(const zisa::array_view<complex_t, dim_v + 1> &u_hat,
                          const zisa::array_view<complex_t, dim_v + 1> &u,
                          int direction)
    : super(direction) {
  initialize(u_hat, u);
}

template <int Dim>
CUFFT_C2C<Dim>::~CUFFT_C2C() {
  if (is_forward()) {
    cufftDestroy(plan_forward_);
  }
  if (is_backward()) {
    cufftDestroy(plan_backward_);
  }
  cudaFree(work_area_);
}

template <int Dim>
void CUFFT_C2C<Dim>::forward() {
  LOG_ERR_IF(!is_forward(), "Forward operation was not initialized");
  AZEBAN_PROFILE_START("CUFFT_C2C::forward");
  auto status = execute_c2c(plan_forward_, u_.raw(), u_hat_.raw(), FFT_FORWARD);
  cudaCheckError(status);
  cudaDeviceSynchronize();
  AZEBAN_PROFILE_STOP("CUFFT_C2C::forward");
}

template <int Dim>
void CUFFT_C2C<Dim>::backward() {
  LOG_ERR_IF(!is_backward(), "Backward operation was not initialized");
  AZEBAN_PROFILE_START("CUFFT_C2C::backward");
  auto status
      = execute_c2c(plan_backward_, u_hat_.raw(), u_.raw(), FFT_BACKWARD);
  cudaCheckError(status);
  cudaDeviceSynchronize();
  AZEBAN_PROFILE_STOP("CUFFT_C2C::backward");
}

template <int Dim>
void CUFFT_C2C<Dim>::do_initialize(
    const zisa::array_view<complex_t, Dim + 1> &u_hat,
    const zisa::array_view<complex_t, Dim + 1> &u) {
  LOG_ERR_IF(u_hat.memory_location() != zisa::device_type::cuda,
             "cuFFT is GPU only!");
  LOG_ERR_IF(u.memory_location() != zisa::device_type::cuda,
             "cuFFT is GPU only!");
  int rdist = 1;
  int cdist = 1;
  int n[dim_v];
  for (int i = 0; i < dim_v; ++i) {
    rdist *= u.shape(i + 1);
    cdist *= u_hat.shape(i + 1);
    n[i] = u.shape(i + 1);
  }
  size_t workspace_size = 0;
  // Create a plan for the forward operation
  if (direction_ & FFT_FORWARD) {
    cufftResult status = cufftCreate(&plan_forward_);
    cudaCheckError(status);
    status = cufftSetAutoAllocation(plan_forward_, false);
    cudaCheckError(status);
    // TODO: Implement
    size_t fwd_size;
    status = cufftGetSize(plan_forward_, &fwd_size);
    cudaCheckError(status);
    workspace_size = std::max(workspace_size, fwd_size);
  }
  // Create a plan for the backward operation
  if (direction_ & FFT_BACKWARD) {
    cufftResult status = cufftCreate(&plan_backward_);
    cudaCheckError(status);
    status = cufftSetAutoAllocation(plan_backward_, false);
    cudaCheckError(status);
    // TODO: Implement
    size_t bkw_size;
    status = cufftGetSize(plan_backward_, &bkw_size);
    cudaCheckError(status);
    workspace_size = std::max(workspace_size, bkw_size);
  }
  // Allocate the shared work area
  cudaMalloc((void **)(&work_area_), workspace_size);
  if (direction_ & FFT_FORWARD) {
    cufftResult status = cufftSetWorkArea(plan_forward_, work_area_);
    cudaCheckError(status);
  }
  if (direction_ & FFT_BACKWARD) {
    cufftResult status = cufftSetWorkArea(plan_backward_, work_area_);
    cudaCheckError(status);
  }
}

template class CUFFT_R2C<1>;
template class CUFFT_R2C<2>;
template class CUFFT_R2C<3>;
template class CUFFT_C2C<1>;
template class CUFFT_C2C<2>;
template class CUFFT_C2C<3>;

}
