#include <azeban/operations/fftwfft.hpp>
#include <azeban/profiler.hpp>

namespace azeban {

static fftwf_plan plan_many_dft(int rank,
                                const int *n,
                                int howmany,
                                Complex<float> *in,
                                const int *inembed,
                                int istride,
                                int idist,
                                Complex<float> *out,
                                const int *onembed,
                                int ostride,
                                int odist,
                                fft_direction direction,
                                unsigned flags) {
  int sign;
  if (direction == FFT_FORWARD) {
    sign = FFTW_FORWARD;
  } else {
    sign = FFTW_BACKWARD;
  }
  return fftwf_plan_many_dft(rank,
                             n,
                             howmany,
                             reinterpret_cast<fftwf_complex *>(in),
                             inembed,
                             istride,
                             idist,
                             reinterpret_cast<fftwf_complex *>(out),
                             onembed,
                             ostride,
                             odist,
                             sign,
                             flags);
}

static fftw_plan plan_many_dft(int rank,
                               const int *n,
                               int howmany,
                               Complex<double> *in,
                               const int *inembed,
                               int istride,
                               int idist,
                               Complex<double> *out,
                               const int *onembed,
                               int ostride,
                               int odist,
                               fft_direction direction,
                               unsigned flags) {
  int sign;
  if (direction == FFT_FORWARD) {
    sign = FFTW_FORWARD;
  } else {
    sign = FFTW_BACKWARD;
  }
  return fftw_plan_many_dft(rank,
                            n,
                            howmany,
                            reinterpret_cast<fftw_complex *>(in),
                            inembed,
                            istride,
                            idist,
                            reinterpret_cast<fftw_complex *>(out),
                            onembed,
                            ostride,
                            odist,
                            sign,
                            flags);
}

static fftwf_plan plan_many_dft_r2c(int rank,
                                    const int *n,
                                    int howmany,
                                    float *in,
                                    const int *inembed,
                                    int istride,
                                    int idist,
                                    Complex<float> *out,
                                    const int *onembed,
                                    int ostride,
                                    int odist,
                                    unsigned flags) {
  return fftwf_plan_many_dft_r2c(rank,
                                 n,
                                 howmany,
                                 in,
                                 inembed,
                                 istride,
                                 idist,
                                 reinterpret_cast<fftwf_complex *>(out),
                                 onembed,
                                 ostride,
                                 odist,
                                 flags);
}

static fftw_plan plan_many_dft_r2c(int rank,
                                   const int *n,
                                   int howmany,
                                   double *in,
                                   const int *inembed,
                                   int istride,
                                   int idist,
                                   Complex<double> *out,
                                   const int *onembed,
                                   int ostride,
                                   int odist,
                                   unsigned flags) {
  return fftw_plan_many_dft_r2c(rank,
                                n,
                                howmany,
                                in,
                                inembed,
                                istride,
                                idist,
                                reinterpret_cast<fftw_complex *>(out),
                                onembed,
                                ostride,
                                odist,
                                flags);
}

static fftwf_plan plan_many_dft_c2r(int rank,
                                    const int *n,
                                    int howmany,
                                    Complex<float> *in,
                                    const int *inembed,
                                    int istride,
                                    int idist,
                                    float *out,
                                    const int *onembed,
                                    int ostride,
                                    int odist,
                                    unsigned flags) {
  return fftwf_plan_many_dft_c2r(rank,
                                 n,
                                 howmany,
                                 reinterpret_cast<fftwf_complex *>(in),
                                 inembed,
                                 istride,
                                 idist,
                                 out,
                                 onembed,
                                 ostride,
                                 odist,
                                 flags);
}

static fftw_plan plan_many_dft_c2r(int rank,
                                   const int *n,
                                   int howmany,
                                   Complex<double> *in,
                                   const int *inembed,
                                   int istride,
                                   int idist,
                                   double *out,
                                   const int *onembed,
                                   int ostride,
                                   int odist,
                                   unsigned flags) {
  return fftw_plan_many_dft_c2r(rank,
                                n,
                                howmany,
                                reinterpret_cast<fftw_complex *>(in),
                                inembed,
                                istride,
                                idist,
                                out,
                                onembed,
                                ostride,
                                odist,
                                flags);
}

static void destroy_plan(fftwf_plan plan) { fftwf_destroy_plan(plan); }

static void destroy_plan(fftw_plan plan) { fftw_destroy_plan(plan); }

static void execute(const fftwf_plan plan) { fftwf_execute(plan); }

static void execute(const fftw_plan plan) { fftw_execute(plan); }

template <>
template <>
FFTWFFT_R2C<1>::FFTWFFT_R2C<true, void>(int direction, bool transform_x)
    : super(direction, transform_x) {}

template <>
template <>
FFTWFFT_R2C<1>::FFTWFFT_R2C<true, void>(
    const zisa::array_view<complex_t, 2> &u_hat,
    const zisa::array_view<real_t, 2> &u,
    int direction,
    bool transform_x)
    : FFTWFFT_R2C(direction, transform_x) {
  initialize(u_hat, u);
}

template <>
template <>
FFTWFFT_R2C<2>::FFTWFFT_R2C<true, void>(int direction,
                                        bool transform_x,
                                        bool transform_y)
    : super(direction, transform_x, transform_y) {}

template <>
template <>
FFTWFFT_R2C<2>::FFTWFFT_R2C<true, void>(
    const zisa::array_view<complex_t, 3> &u_hat,
    const zisa::array_view<real_t, 3> &u,
    int direction,
    bool transform_x,
    bool transform_y)
    : FFTWFFT_R2C(direction, transform_x, transform_y) {
  initialize(u_hat, u);
}

template <>
template <>
FFTWFFT_R2C<3>::FFTWFFT_R2C<true, void>(int direction,
                                        bool transform_x,
                                        bool transform_y,
                                        bool transform_z)
    : super(direction, transform_x, transform_y, transform_z) {}

template <>
template <>
FFTWFFT_R2C<3>::FFTWFFT_R2C<true, void>(
    const zisa::array_view<complex_t, 4> &u_hat,
    const zisa::array_view<real_t, 4> &u,
    int direction,
    bool transform_x,
    bool transform_y,
    bool transform_z)
    : FFTWFFT_R2C(direction, transform_x, transform_y, transform_z) {
  initialize(u_hat, u);
}

template <int Dim>
FFTWFFT_R2C<Dim>::~FFTWFFT_R2C() {
  if (is_forward()) {
    destroy_plan(plan_forward_);
  }
  if (is_backward()) {
    destroy_plan(plan_backward_);
  }
}

template <int Dim>
void FFTWFFT_R2C<Dim>::forward() {
  LOG_ERR_IF(!is_forward(), "Forward operation was not initialized");
  AZEBAN_PROFILE_START("FFTWFFT_R2C::forward");
  execute(plan_forward_);
  AZEBAN_PROFILE_STOP("FFTWFFT_R2C::forward");
}

template <int Dim>
void FFTWFFT_R2C<Dim>::backward() {
  LOG_ERR_IF(!is_backward(), "Backward operation was not initialized");
  AZEBAN_PROFILE_START("FFTWFFT_R2C::backward");
  execute(plan_backward_);
  AZEBAN_PROFILE_STOP("FFTWFFT_R2C::backward");
}

template <int Dim>
void FFTWFFT_R2C<Dim>::do_initialize(
    const zisa::array_view<complex_t, Dim + 1> &u_hat,
    const zisa::array_view<real_t, Dim + 1> &u) {
  LOG_ERR_IF(u_hat.memory_location() != zisa::device_type::cpu,
             "FFTW is CPU only!");
  LOG_ERR_IF(u.memory_location() != zisa::device_type::cpu,
             "FFTW is CPU only!");

  int rank = 0;
  for (int i = 0; i < Dim; ++i) {
    rank += transform_dims_[i] ? 1 : 0;
  }

  std::vector<int> n;
  for (int i = 0; i < Dim; ++i) {
    if (transform_dims_[i]) {
      n.push_back(u.shape(i + 1));
    }
  }

  int howmany = data_dim_;
  for (int i = 0; i < Dim; ++i) {
    if (!transform_dims_[i]) {
      howmany *= u.shape(i + 1);
    }
  }

  int rstride = 1;
  int cstride = 1;
  for (int i = Dim - 1; i >= 0; --i) {
    if (transform_dims_[i]) {
      break;
    }
    rstride *= u.shape(i + 1);
    cstride *= u_hat.shape(i + 1);
  }

  int rdist = 1;
  int cdist = 1;
  for (int i = 0; i < dim_v; ++i) {
    if (!transform_dims_[i]) {
      break;
    }
    rdist *= u.shape(i + 1);
    cdist *= u_hat.shape(i + 1);
  }

  if (direction_ & FFT_FORWARD) {
    // Create a plan for the forward operation
    plan_forward_ = plan_many_dft_r2c(rank,          // rank
                                      n.data(),      // n
                                      howmany,       // howmany
                                      u.raw(),       // in
                                      NULL,          // inembed
                                      rstride,       // istride
                                      rdist,         // idist
                                      u_hat.raw(),   // out
                                      NULL,          // onembed
                                      cstride,       // ostride
                                      cdist,         // odist
                                      FFTW_MEASURE); // flags
  }
  if (direction_ & FFT_BACKWARD) {
    // Create a plan for the backward operation
    plan_backward_ = plan_many_dft_c2r(rank,          // rank
                                       n.data(),      // n
                                       howmany,       // howmany
                                       u_hat.raw(),   // in
                                       NULL,          // inembed
                                       cstride,       // istride
                                       cdist,         // idist
                                       u.raw(),       // out
                                       NULL,          // onembed
                                       rstride,       // ostride
                                       rdist,         // odist
                                       FFTW_MEASURE); // flags
  }
}

template <>
template <>
FFTWFFT_C2C<1>::FFTWFFT_C2C<true, void>(int direction, bool transform_x)
    : super(direction, transform_x) {}

template <>
template <>
FFTWFFT_C2C<1>::FFTWFFT_C2C<true, void>(
    const zisa::array_view<complex_t, 2> &u_hat,
    const zisa::array_view<complex_t, 2> &u,
    int direction,
    bool transform_x)
    : FFTWFFT_C2C(direction, transform_x) {
  initialize(u_hat, u);
}

template <>
template <>
FFTWFFT_C2C<2>::FFTWFFT_C2C<true, void>(int direction,
                                        bool transform_x,
                                        bool transform_y)
    : super(direction, transform_x, transform_y) {}

template <>
template <>
FFTWFFT_C2C<2>::FFTWFFT_C2C<true, void>(
    const zisa::array_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &u,
    int direction,
    bool transform_x,
    bool transform_y)
    : FFTWFFT_C2C(direction, transform_x, transform_y) {
  initialize(u_hat, u);
}

template <>
template <>
FFTWFFT_C2C<3>::FFTWFFT_C2C<true, void>(int direction,
                                        bool transform_x,
                                        bool transform_y,
                                        bool transform_z)
    : super(direction, transform_x, transform_y, transform_z) {}

template <>
template <>
FFTWFFT_C2C<3>::FFTWFFT_C2C<true, void>(
    const zisa::array_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &u,
    int direction,
    bool transform_x,
    bool transform_y,
    bool transform_z)
    : FFTWFFT_C2C(direction, transform_x, transform_y, transform_z) {
  initialize(u_hat, u);
}

template <int Dim>
FFTWFFT_C2C<Dim>::~FFTWFFT_C2C() {
  if (is_forward()) {
    destroy_plan(plan_forward_);
  }
  if (is_backward()) {
    destroy_plan(plan_backward_);
  }
}

template <int Dim>
void FFTWFFT_C2C<Dim>::forward() {
  LOG_ERR_IF(!is_forward(), "Forward operation was not initialized");
  AZEBAN_PROFILE_START("FFTWFFT_C2C::forward");
  execute(plan_forward_);
  AZEBAN_PROFILE_STOP("FFTWFFT_C2C::forward");
}

template <int Dim>
void FFTWFFT_C2C<Dim>::backward() {
  LOG_ERR_IF(!is_backward(), "Backward operation was not initialized");
  AZEBAN_PROFILE_START("FFTWFFT_C2C::backward");
  execute(plan_backward_);
  AZEBAN_PROFILE_STOP("FFTWFFT_C2C::backward");
}

template <int Dim>
void FFTWFFT_C2C<Dim>::do_initialize(
    const zisa::array_view<complex_t, Dim + 1> &u_hat,
    const zisa::array_view<complex_t, Dim + 1> &u) {
  LOG_ERR_IF(u_hat.memory_location() != zisa::device_type::cpu,
             "FFTW is CPU only!");
  LOG_ERR_IF(u.memory_location() != zisa::device_type::cpu,
             "FFTW is CPU only!");

  int rank = 0;
  for (int i = 0; i < Dim; ++i) {
    rank += transform_dims_[i] ? 1 : 0;
  }

  std::vector<int> n;
  for (int i = 0; i < Dim; ++i) {
    if (transform_dims_[i]) {
      n.push_back(u.shape(i + 1));
    }
  }

  int howmany = data_dim_;
  for (int i = 0; i < Dim; ++i) {
    if (!transform_dims_[i]) {
      howmany *= u.shape(i + 1);
    }
  }

  int rstride = 1;
  int cstride = 1;
  for (int i = Dim - 1; i >= 0; --i) {
    if (transform_dims_[i]) {
      break;
    }
    rstride *= u.shape(i + 1);
    cstride *= u_hat.shape(i + 1);
  }

  int rdist = 1;
  int cdist = 1;
  for (int i = 0; i < dim_v; ++i) {
    if (!transform_dims_[i]) {
      break;
    }
    rdist *= u.shape(i + 1);
    cdist *= u_hat.shape(i + 1);
  }

  if (direction_ & FFT_FORWARD) {
    // Create a plan for the forward operation
    plan_many_dft(rank,
                  n.data(),
                  howmany,
                  u.raw(),
                  NULL,
                  rstride,
                  rdist,
                  u_hat.raw(),
                  NULL,
                  cstride,
                  cdist,
                  FFT_FORWARD,
                  FFTW_MEASURE);
  }
  if (direction_ & FFT_BACKWARD) {
    // Create a plan for the backward operation
    plan_many_dft(rank,
                  n.data(),
                  howmany,
                  u_hat.raw(),
                  NULL,
                  cstride,
                  cdist,
                  u.raw(),
                  NULL,
                  rstride,
                  rdist,
                  FFT_BACKWARD,
                  FFTW_MEASURE);
  }
}

template class FFTWFFT_R2C<1>;
template class FFTWFFT_R2C<2>;
template class FFTWFFT_R2C<3>;
template class FFTWFFT_C2C<1>;
template class FFTWFFT_C2C<2>;
template class FFTWFFT_C2C<3>;

}
