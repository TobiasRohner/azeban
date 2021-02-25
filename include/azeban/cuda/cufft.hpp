#ifndef CUFFT_H_
#define CUFFT_H_

#include <azeban/fft_base.hpp>
#include <cufft.h>

namespace azeban {


template<int Dim>
class CUFFT final : public FFT<Dim> {
  using super = FFT<Dim>;

  static_assert(std::is_same_v<real_t, float> || std::is_same_v<real_t, double>,
		"Only single and double precision are supported by CUFFT");

public:
  static constexpr int dim_v = Dim;

  CUFFT(const zisa::array_view<complex_t, dim_v> &u_hat, const zisa::array_view<real_t, dim_v> &u)
    : CUFFT(increase_array_view_rank(u_hat), increase_array_view_rank(u)) { }

  CUFFT(const zisa::array_view<complex_t, dim_v+1> &u_hat, const zisa::array_view<real_t, dim_v+1> &u)
      : super(u_hat, u) {
    assert(u_hat.memory_location() == zisa::device_type::cuda && "cuFFT is GPU only!");
    assert(u.memory_location() == zisa::device_type::cuda && "cuFFT is GPU only!");
    // Create a plan for the forward operation
    int rdist = 1;
    int cdist = 1;
    int n[dim_v];
    for (int i = 0 ; i < dim_v ; ++i) {
      rdist *= u_.shape()[i+1];
      cdist *= u_hat_.shape()[i+1];
      n[i] = u_.shape()[i+1];
    }
    auto status = cufftPlanMany(&plan_forward_,
		                dim_v,
		                n,
		                n,
		                1,
		                rdist,
		                n,
		                1,
		                cdist,
				type_forward,
				1);
    assert(status == CUFFT_SUCCESS);
    // Create a plan for the backward operation
    status = cufftPlanMany(&plan_backward_,
			   dim_v,
			   n,
			   n,
			   1,
			   cdist,
			   n,
			   1,
			   rdist,
			   type_backward,
			   1);
  }

  virtual ~CUFFT() override { }

  virtual void forward() override {
    if constexpr (std::is_same_v<float, real_t>) {
      auto status = cufftExecR2C(plan_forward_, u_.raw(), reinterpret_cast<cufftComplex*>(u_hat_.raw()));
      assert(status == CUFFT_SUCCESS);
    }
    else {
      auto status = cufftExecD2Z(plan_forward_, u_.raw(), reinterpret_cast<cufftDoubleComplex*>(u_hat_.raw()));
      assert(status == CUFFT_SUCCESS);
    }
  }

  virtual void backward() override {
    if constexpr (std::is_same_v<float, real_t>) {
      auto status = cufftExecC2R(plan_backward_, reinterpret_cast<cufftComplex*>(u_hat_.raw()), u_.raw());
      assert(status == CUFFT_SUCCESS);
    }
    else {
      auto status = cufftExecZ2D(plan_backward_, reinterpret_cast<cufftDoubleComplex*>(u_hat_.raw()), u_.raw());
      assert(status == CUFFT_SUCCESS);
    }
  }

protected:
  using super::u_hat_;
  using super::u_;
  using super::data_dim_;
  using super::increase_array_view_rank;

private:
  cufftHandle plan_forward_;
  cufftHandle plan_backward_;

  static constexpr cufftType type_forward = std::is_same_v<float, real_t> ? CUFFT_R2C : CUFFT_D2Z;
  static constexpr cufftType type_backward = std::is_same_v<float, real_t> ? CUFFT_C2R : CUFFT_Z2D;
};


}



#endif
