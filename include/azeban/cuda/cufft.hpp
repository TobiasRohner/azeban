#ifndef CUFFT_H_
#define CUFFT_H_

#include <azeban/cuda/cuda_check_error.hpp>
#include <azeban/fft_base.hpp>
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
        const zisa::array_view<real_t, dim_v + 1> &u)
      : super(u_hat, u) {
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
    // Create a plan for the backward operation
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
  }

  virtual ~CUFFT() override {
    cufftDestroy(plan_forward_);
    cufftDestroy(plan_backward_);
  }

  virtual void forward() override {
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
  }

  virtual void backward() override {
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
  }

protected:
  using super::data_dim_;
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
