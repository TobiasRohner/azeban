#ifndef CUFFT_H_
#define CUFFT_H_

#include <azeban/cuda/cuda_check_error.hpp>
#include <azeban/operations/fft_base.hpp>
#include <azeban/profiler.hpp>
#include <cufft.h>

namespace azeban {

template <int Dim>
class CUFFT final : public FFT<Dim> {
  using super = FFT<Dim>;

  static_assert(std::is_same_v<real_t, float> || std::is_same_v<real_t, double>,
                "Only single and double precision are supported by CUFFT");

public:
  static constexpr int dim_v = Dim;

  CUFFT(const zisa::array_view<complex_t, dim_v + 1> &u_hat,
        const zisa::array_view<real_t, dim_v + 1> &u,
        int direction = FFT_FORWARD | FFT_BACKWARD)
      : super(u_hat, u, direction) {
    assert(u_hat.memory_location() == zisa::device_type::cuda
           && "cuFFT is GPU only!");
    assert(u.memory_location() == zisa::device_type::cuda
           && "cuFFT is GPU only!");
    // Create a plan for the forward operation
    int rdist = 1;
    int cdist = 1;
    int n[dim_v];
    for (int i = 0; i < dim_v; ++i) {
      rdist *= u_.shape(i + 1);
      cdist *= u_hat_.shape(i + 1);
      n[i] = u_.shape(i + 1);
    }
    if (direction_ & FFT_FORWARD) {
      auto status = cufftPlanMany(&plan_forward_, // plan
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
    }
    // Create a plan for the backward operation
    if (direction_ & FFT_BACKWARD) {
      auto status = cufftPlanMany(&plan_backward_, // plan
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
    }
  }

  virtual ~CUFFT() override {
    if (direction_ & FFT_FORWARD) {
      cufftDestroy(plan_forward_);
    }
    if (direction_ & FFT_BACKWARD) {
      cufftDestroy(plan_backward_);
    }
  }

  virtual void forward() override {
    LOG_ERR_IF((direction_ & FFT_FORWARD) == 0,
               "Forward operation was not initialized");
    AZEBAN_PROFILE_START("CUFFT::forward");
    if constexpr (std::is_same_v<float, real_t>) {
      auto status
          = cufftExecR2C(plan_forward_,
                         u_.raw(),
                         reinterpret_cast<cufftComplex *>(u_hat_.raw()));
      cudaCheckError(status);
    } else {
      auto status
          = cufftExecD2Z(plan_forward_,
                         u_.raw(),
                         reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()));
      cudaCheckError(status);
    }
    AZEBAN_PROFILE_STOP("CUFFT::forward");
  }

  virtual void backward() override {
    LOG_ERR_IF((direction_ & FFT_BACKWARD) == 0,
               "Backward operation was not initialized");
    AZEBAN_PROFILE_START("CUFFT::backward");
    if constexpr (std::is_same_v<float, real_t>) {
      auto status = cufftExecC2R(plan_backward_,
                                 reinterpret_cast<cufftComplex *>(u_hat_.raw()),
                                 u_.raw());
      cudaCheckError(status);
    } else {
      auto status
          = cufftExecZ2D(plan_backward_,
                         reinterpret_cast<cufftDoubleComplex *>(u_hat_.raw()),
                         u_.raw());
      cudaCheckError(status);
    }
    AZEBAN_PROFILE_STOP("CUFFT::backward");
  }

protected:
  using super::data_dim_;
  using super::direction_;
  using super::u_;
  using super::u_hat_;

private:
  cufftHandle plan_forward_;
  cufftHandle plan_backward_;

  static constexpr cufftType type_forward
      = std::is_same_v<float, real_t> ? CUFFT_R2C : CUFFT_D2Z;
  static constexpr cufftType type_backward
      = std::is_same_v<float, real_t> ? CUFFT_C2R : CUFFT_Z2D;
};

}

#endif
