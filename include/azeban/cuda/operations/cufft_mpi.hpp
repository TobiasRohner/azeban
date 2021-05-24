#ifndef CUFFT_MPI_H_
#define CUFFT_MPI_H_

#include <azeban/cuda/cuda_check_error.hpp>
#include <azeban/operations/fft_base.hpp>
#include <azeban/profiler.hpp>
#include <cufft.h>
#include <mpi.h>
#include <vector>

namespace azeban {

// Assumes array is split between ranks in the first dimension
template <int Dim>
class CUFFT_MPI final : public FFT<Dim> {
  static_assert(Dim == 2 || Dim == 3,
                "CUFFT_MPI only works with 2D or 3D data");
};

template <>
class CUFFT_MPI<2> final : public FFT<2> {
  using super = FFT<2>;

  static_assert(std::is_same_v<real_t, float> || std::is_same_v<real_t, double>,
                "Only single and double precision are supported by CUFFT");

public:
  static constexpr int dim_v = 2;

  CUFFT_MPI(const zisa::array_view<complex_t, 3> &u_hat,
            const zisa::array_view<real_t, 3> &u,
            MPI_Comm comm,
            int direction = FFT_FORWARD | FFT_BACKWARD);

  virtual ~CUFFT_MPI() override;

  virtual void forward() override;
  virtual void backward() override;

protected:
  using super::data_dim_;
  using super::direction_;
  using super::u_;
  using super::u_hat_;

private:
  cufftHandle plan_forward_r2c_;
  cufftHandle plan_forward_c2c_;
  cufftHandle plan_backward_c2r_;
  cufftHandle plan_backward_c2c_;
  MPI_Comm comm_;
  void *work_area_;
  zisa::array<complex_t, 3> partial_u_hat_;
  zisa::array<complex_t, 3> mpi_send_buffer_;
  zisa::array<complex_t, 3> mpi_recv_buffer_;

  static constexpr cufftType type_forward_r2c
      = std::is_same_v<float, real_t> ? CUFFT_R2C : CUFFT_D2Z;
  static constexpr cufftType type_backward_c2r
      = std::is_same_v<float, real_t> ? CUFFT_C2R : CUFFT_Z2D;
  static constexpr cufftType type_forward_c2c
      = std::is_same_v<float, real_t> ? CUFFT_C2C : CUFFT_Z2Z;
  static constexpr cufftType type_backward_c2c
      = std::is_same_v<float, real_t> ? CUFFT_C2C : CUFFT_Z2Z;
};

template <>
class CUFFT_MPI<3> final : public FFT<3> {
  using super = FFT<3>;

  static_assert(std::is_same_v<real_t, float> || std::is_same_v<real_t, double>,
                "Only single and double precision are supported by CUFFT");

public:
  static constexpr int dim_v = 3;

  CUFFT_MPI(const zisa::array_view<complex_t, 4> &u_hat,
            const zisa::array_view<real_t, 4> &u,
            MPI_Comm comm,
            int direction = FFT_FORWARD | FFT_BACKWARD);

  virtual ~CUFFT_MPI() override;

  virtual void forward() override;
  virtual void backward() override;

protected:
  using super::data_dim_;
  using super::direction_;
  using super::u_;
  using super::u_hat_;

private:
  cufftHandle plan_forward_r2c_;
  cufftHandle plan_forward_c2c_;
  cufftHandle plan_backward_c2r_;
  cufftHandle plan_backward_c2c_;
  MPI_Comm comm_;
  void *work_area_;
  zisa::array<complex_t, 4> partial_u_hat_;
  zisa::array<complex_t, 4> mpi_send_buffer_;
  zisa::array<complex_t, 4> mpi_recv_buffer_;

  static constexpr cufftType type_forward_r2c
      = std::is_same_v<float, real_t> ? CUFFT_R2C : CUFFT_D2Z;
  static constexpr cufftType type_backward_c2r
      = std::is_same_v<float, real_t> ? CUFFT_C2R : CUFFT_Z2D;
  static constexpr cufftType type_forward_c2c
      = std::is_same_v<float, real_t> ? CUFFT_C2C : CUFFT_Z2Z;
  static constexpr cufftType type_backward_c2c
      = std::is_same_v<float, real_t> ? CUFFT_C2C : CUFFT_Z2Z;
};

}

#endif
