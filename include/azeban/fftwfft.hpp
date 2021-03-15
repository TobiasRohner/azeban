#ifndef FFTWFFT_H_
#define FFTWFFT_H_

#include <azeban/fft_base.hpp>
#include <fftw3.h>

namespace azeban {

template <int Dim>
class FFTWFFT final : public FFT<Dim> {
  using super = FFT<Dim>;

  static_assert(
      std::is_same_v<real_t,
                     std::decay_t<decltype(std::declval<fftw_complex>()[0])>>,
      "Real type has the wrong precision for FFTW");
  static_assert(
      std::is_same_v<std::decay_t<decltype(std::declval<complex_t>().x)>,
                     std::decay_t<decltype(std::declval<fftw_complex>()[0])>>,
      "Complex type has the wrong precision for FFTW");

public:
  static constexpr int dim_v = Dim;

  FFTWFFT(const zisa::array_view<complex_t, dim_v + 1> &u_hat,
          const zisa::array_view<real_t, dim_v + 1> &u)
      : super(u_hat, u) {
    assert(u_hat.memory_location() == zisa::device_type::cpu
           && "FFTW is CPU only!");
    assert(u.memory_location() == zisa::device_type::cpu
           && "FFTW is CPU only!");
    // Create a plan for the forward operation
    int rdist = 1;
    int cdist = 1;
    int n[dim_v];
    for (int i = 0; i < dim_v; ++i) {
      rdist *= u_.shape()[i + 1];
      cdist *= u_hat_.shape()[i + 1];
      n[i] = u_.shape()[i + 1];
    }
    plan_forward_ = fftw_plan_many_dft_r2c(
        dim_v,                                        // rank
        n,                                            // n
        data_dim_,                                    // howmany
        u.raw(),                                      // in
        NULL,                                         // inembed
        1,                                            // istride
        rdist,                                        // idist
        reinterpret_cast<real_t(*)[2]>(u_hat_.raw()), // out
        NULL,                                         // onembed
        1,                                            // ostride
        cdist,                                        // odist
        FFTW_MEASURE);                                // flags
    // Create a plan for the backward operation
    plan_backward_ = fftw_plan_many_dft_c2r(
        dim_v,                                        // rank
        n,                                            // n
        data_dim_,                                    // howmany
        reinterpret_cast<real_t(*)[2]>(u_hat_.raw()), // in
        NULL,                                         // inembed
        1,                                            // istride
        cdist,                                        // idist
        u_.raw(),                                     // out
        NULL,                                         // onembed
        1,                                            // ostride
        rdist,                                        // odist
        FFTW_MEASURE);                                // flags
  }

  virtual ~FFTWFFT() override {
    fftw_destroy_plan(plan_forward_);
    fftw_destroy_plan(plan_backward_);
  }

  virtual void forward() override { fftw_execute(plan_forward_); }

  virtual void backward() override { fftw_execute(plan_backward_); }

protected:
  using super::data_dim_;
  using super::u_;
  using super::u_hat_;

private:
  fftw_plan plan_forward_;
  fftw_plan plan_backward_;
};

}

#endif
