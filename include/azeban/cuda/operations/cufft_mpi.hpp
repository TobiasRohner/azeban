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
#ifndef CUFFT_MPI_H_
#define CUFFT_MPI_H_

#include <azeban/cuda/cuda_check_error.hpp>
#include <azeban/operations/fft.hpp>
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
            int direction = FFT_FORWARD | FFT_BACKWARD,
            void *work_area = nullptr);

  virtual ~CUFFT_MPI() override;

  using super::initialize;

  virtual void forward() override;
  virtual void backward() override;

  virtual void *get_work_area() const override;

protected:
  using super::direction_;
  using super::u_;
  using super::u_hat_;

  virtual void do_initialize(const zisa::array_view<complex_t, 3> &u_hat,
                             const zisa::array_view<real_t, 3> &u) override;

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
  bool free_work_area_;

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
            int direction = FFT_FORWARD | FFT_BACKWARD,
            void *work_area = nullptr);

  virtual ~CUFFT_MPI() override;

  using super::initialize;

  virtual void forward() override;
  virtual void backward() override;

  virtual void *get_work_area() const override;

protected:
  using super::direction_;
  using super::u_;
  using super::u_hat_;

  virtual void do_initialize(const zisa::array_view<complex_t, 4> &u_hat,
                             const zisa::array_view<real_t, 4> &u) override;

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
  bool free_work_area_;

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
