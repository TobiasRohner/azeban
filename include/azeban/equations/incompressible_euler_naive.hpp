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
#ifndef INCOMPRESSIBLE_EULER_NAIVE_H_
#define INCOMPRESSIBLE_EULER_NAIVE_H_

#include "equation.hpp"
#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <azeban/operations/convolve.hpp>
#include <azeban/operations/fft_factory.hpp>
#include <azeban/operations/norm.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/equations/incompressible_euler_naive_cuda.hpp>
#endif
#include <azeban/profiler.hpp>

namespace azeban {

template <int Dim, typename SpectralViscosity>
class IncompressibleEulerNaive final : public Equation<Dim> {
  using super = Equation<Dim>;
  static_assert(Dim == 2 || Dim == 3,
                "Incompressible Euler is only implemented for 2D and 3D");

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = Dim;

  IncompressibleEulerNaive(const Grid<dim_v> &grid,
                           const SpectralViscosity &visc,
                           zisa::device_type device)
      : super(grid), device_(device), u_max_(0), visc_(visc) {
    u_hat_ = grid.make_array_fourier_pad(dim_v, device);
    u_ = grid.make_array_phys_pad(dim_v, device);
    B_hat_ = grid.make_array_fourier_pad((dim_v * dim_v + dim_v) / 2, device);
    B_ = grid.make_array_phys_pad((dim_v * dim_v + dim_v) / 2, device);
    fft_u_ = make_fft<dim_v>(u_hat_, u_);
    fft_B_ = make_fft<dim_v>(B_hat_, B_);
  }
  IncompressibleEulerNaive(const IncompressibleEulerNaive &) = delete;
  IncompressibleEulerNaive(IncompressibleEulerNaive &&) = default;
  virtual ~IncompressibleEulerNaive() = default;
  IncompressibleEulerNaive &operator=(const IncompressibleEulerNaive &)
      = delete;
  IncompressibleEulerNaive &operator=(IncompressibleEulerNaive &&) = default;

  virtual void dudt(const zisa::array_view<complex_t, dim_v + 1> &dudt_hat,
                    const zisa::array_const_view<complex_t, dim_v + 1> &u_hat,
                    double,
                    double,
                    double) override {
    ProfileHost profile("IncompressibleEulerNaive::dudt");
    for (int i = 0; i < dim_v; ++i) {
      copy_to_padded(component(u_hat_, i), component(u_hat, i));
    }
    fft_u_->backward();
    u_max_ = max_norm(u_);
    computeB();
    fft_B_->forward();
    if (device_ == zisa::device_type::cpu) {
      if constexpr (dim_v == 2) {
        for (int i = 0; i < zisa::integer_cast<int>(u_hat.shape(1)); ++i) {
          const int i_B = i >= zisa::integer_cast<int>(u_hat.shape(1) / 2 + 1)
                              ? B_hat_.shape(1) - u_hat.shape(1) + i
                              : i;
          for (int j = 0; j < zisa::integer_cast<int>(u_hat.shape(2)); ++j) {
            int i_ = i;
            if (i >= zisa::integer_cast<int>(u_hat.shape(1)) / 2 + 1) {
              i_ -= u_hat.shape(1);
            }
            const real_t k1 = 2 * zisa::pi * i_;
            const real_t k2 = 2 * zisa::pi * j;
            const complex_t B11_hat = B_hat_(0, i_B, j);
            const complex_t B12_hat = B_hat_(1, i_B, j);
            const complex_t B22_hat = B_hat_(2, i_B, j);
            const complex_t b1_hat
                = complex_t(0, k1) * B11_hat + complex_t(0, k2) * B12_hat;
            const complex_t b2_hat
                = complex_t(0, k1) * B12_hat + complex_t(0, k2) * B22_hat;
            const real_t absk2 = k1 * k1 + k2 * k2;
            const complex_t L1_hat = (1. - (k1 * k1) / absk2) * b1_hat
                                     + (0. - (k1 * k2) / absk2) * b2_hat;
            const complex_t L2_hat = (0. - (k2 * k1) / absk2) * b1_hat
                                     + (1. - (k2 * k2) / absk2) * b2_hat;
            const real_t v = visc_.eval(zisa::sqrt(absk2));
            dudt_hat(0, i, j) = absk2 == 0 ? 0 : -L1_hat + v * u_hat(0, i, j);
            dudt_hat(1, i, j) = absk2 == 0 ? 0 : -L2_hat + v * u_hat(1, i, j);
          }
        }
      } else {
        for (int i = 0; i < zisa::integer_cast<int>(u_hat.shape(1)); ++i) {
          const int i_B = i >= zisa::integer_cast<int>(u_hat.shape(1) / 2 + 1)
                              ? B_hat_.shape(1) - u_hat.shape(1) + i
                              : i;
          for (int j = 0; j < zisa::integer_cast<int>(u_hat.shape(2)); ++j) {
            const int j_B = j >= zisa::integer_cast<int>(u_hat.shape(2) / 2 + 1)
                                ? B_hat_.shape(2) - u_hat.shape(2) + j
                                : j;
            for (int k = 0; k < zisa::integer_cast<int>(u_hat.shape(3)); ++k) {
              int i_ = i;
              int j_ = j;
              if (i_ >= zisa::integer_cast<int>(u_hat.shape(1) / 2 + 1)) {
                i_ -= u_hat.shape(1);
              }
              if (j_ >= zisa::integer_cast<int>(u_hat.shape(2) / 2 + 1)) {
                j_ -= u_hat.shape(2);
              }
              const real_t k1 = 2 * zisa::pi * i_;
              const real_t k2 = 2 * zisa::pi * j_;
              const real_t k3 = 2 * zisa::pi * k;
              const complex_t B11_hat = B_hat_(0, i_B, j_B, k);
              const complex_t B21_hat = B_hat_(1, i_B, j_B, k);
              const complex_t B22_hat = B_hat_(2, i_B, j_B, k);
              const complex_t B31_hat = B_hat_(3, i_B, j_B, k);
              const complex_t B32_hat = B_hat_(4, i_B, j_B, k);
              const complex_t B33_hat = B_hat_(5, i_B, j_B, k);
              const complex_t b1_hat = complex_t(0, k1) * B11_hat
                                       + complex_t(0, k2) * B21_hat
                                       + complex_t(0, k3) * B31_hat;
              const complex_t b2_hat = complex_t(0, k1) * B21_hat
                                       + complex_t(0, k2) * B22_hat
                                       + complex_t(0, k3) * B32_hat;
              const complex_t b3_hat = complex_t(0, k1) * B31_hat
                                       + complex_t(0, k2) * B32_hat
                                       + complex_t(0, k3) * B33_hat;
              const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;
              const complex_t L1_hat = (1. - (k1 * k1) / absk2) * b1_hat
                                       + (0. - (k1 * k2) / absk2) * b2_hat
                                       + (0. - (k1 * k3) / absk2) * b3_hat;
              const complex_t L2_hat = (0. - (k2 * k1) / absk2) * b1_hat
                                       + (1. - (k2 * k2) / absk2) * b2_hat
                                       + (0. - (k2 * k3) / absk2) * b3_hat;
              const complex_t L3_hat = (0. - (k3 * k1) / absk2) * b1_hat
                                       + (0. - (k3 * k2) / absk2) * b2_hat
                                       + (1. - (k3 * k3) / absk2) * b3_hat;
              const real_t v = visc_.eval(zisa::sqrt(absk2));
              dudt_hat(0, i, j, k)
                  = absk2 == 0 ? 0 : -L1_hat + v * u_hat(0, i, j, k);
              dudt_hat(1, i, j, k)
                  = absk2 == 0 ? 0 : -L2_hat + v * u_hat(1, i, j, k);
              dudt_hat(2, i, j, k)
                  = absk2 == 0 ? 0 : -L3_hat + v * u_hat(2, i, j, k);
            }
          }
        }
      }
    }
#if ZISA_HAS_CUDA
    else if (device_ == zisa::device_type::cuda) {
      if constexpr (dim_v == 2) {
        incompressible_euler_naive_2d_cuda(B_hat_, u_hat, dudt_hat, visc_);
      } else {
        incompressible_euler_naive_3d_cuda(B_hat_, u_hat, dudt_hat, visc_);
      }
    }
#endif
    else {
      LOG_ERR("Unsupported memory_location");
    }
  }

  virtual double dt(double C) const override {
    return C * zisa::pow<Dim - 1>(grid_.N_phys) / u_max_;
  }

  virtual int n_vars() const override { return dim_v; }

protected:
  using super::grid_;

private:
  zisa::device_type device_;
  real_t u_max_;
  zisa::array<complex_t, dim_v + 1> u_hat_;
  zisa::array<real_t, dim_v + 1> u_;
  zisa::array<complex_t, dim_v + 1> B_hat_;
  zisa::array<real_t, dim_v + 1> B_;
  std::shared_ptr<FFT<dim_v>> fft_u_;
  std::shared_ptr<FFT<dim_v>> fft_B_;
  SpectralViscosity visc_;

  template <typename Scalar>
  static zisa::array_view<Scalar, dim_v>
  component(const zisa::array_view<Scalar, dim_v + 1> &arr, int dim) {
    zisa::shape_t<dim_v> shape;
    for (int i = 0; i < dim_v; ++i) {
      shape[i] = arr.shape(i + 1);
    }
    return {
        shape, arr.raw() + dim * zisa::product(shape), arr.memory_location()};
  }

  template <typename Scalar>
  static zisa::array_const_view<Scalar, dim_v>
  component(const zisa::array_const_view<Scalar, dim_v + 1> &arr, int dim) {
    zisa::shape_t<dim_v> shape;
    for (int i = 0; i < dim_v; ++i) {
      shape[i] = arr.shape(i + 1);
    }
    return {
        shape, arr.raw() + dim * zisa::product(shape), arr.memory_location()};
  }

  template <typename Scalar>
  static zisa::array_view<Scalar, dim_v>
  component(zisa::array<Scalar, dim_v + 1> &arr, int dim) {
    zisa::shape_t<dim_v> shape;
    for (int i = 0; i < dim_v; ++i) {
      shape[i] = arr.shape(i + 1);
    }
    return {shape, arr.raw() + dim * zisa::product(shape), arr.device()};
  }

  template <typename Scalar>
  static zisa::array_const_view<Scalar, dim_v>
  component(const zisa::array<Scalar, dim_v + 1> &arr, int dim) {
    zisa::shape_t<dim_v> shape;
    for (int i = 0; i < dim_v; ++i) {
      shape[i] = arr.shape(i + 1);
    }
    return {shape, arr.raw() + dim * zisa::product(shape), arr.device()};
  }

  void computeB() {
    ProfileHost profile("IncompressibleEulerNaive::computeB");
    if (device_ == zisa::device_type::cpu) {
      const real_t norm = 1.0
                          / (zisa::pow<dim_v>(grid_.N_phys)
                             * zisa::pow<dim_v>(grid_.N_phys_pad));
      if constexpr (dim_v == 2) {
        for (zisa::int_t i = 0; i < u_.shape(1); ++i) {
          for (zisa::int_t j = 0; j < u_.shape(2); ++j) {
            const real_t u1 = u_(0, i, j);
            const real_t u2 = u_(1, i, j);
            B_(0, i, j) = norm * u1 * u1;
            B_(1, i, j) = norm * u1 * u2;
            B_(2, i, j) = norm * u2 * u2;
          }
        }
      } else {
        for (zisa::int_t i = 0; i < u_.shape(1); ++i) {
          for (zisa::int_t j = 0; j < u_.shape(2); ++j) {
            for (zisa::int_t k = 0; k < u_.shape(3); ++k) {
              const real_t u1 = u_(0, i, j, k);
              const real_t u2 = u_(1, i, j, k);
              const real_t u3 = u_(2, i, j, k);
              B_(0, i, j, k) = norm * u1 * u1;
              B_(1, i, j, k) = norm * u2 * u1;
              B_(2, i, j, k) = norm * u2 * u2;
              B_(3, i, j, k) = norm * u3 * u1;
              B_(4, i, j, k) = norm * u3 * u2;
              B_(5, i, j, k) = norm * u3 * u3;
            }
          }
        }
      }
    }
#if ZISA_HAS_CUDA
    else if (device_ == zisa::device_type::cuda) {
      incompressible_euler_naive_compute_B_cuda<dim_v>(
          fft_B_->u(), fft_u_->u(), grid_);
    }
#endif
    else {
      LOG_ERR("Unsupported memory location");
    }
  }
};

}

#endif
