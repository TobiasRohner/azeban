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
#ifndef INCOMPRESSIBLE_EULER_H_
#define INCOMPRESSIBLE_EULER_H_

#include "advection_functions.hpp"
#include "equation.hpp"
#include "incompressible_euler_functions.hpp"
#include <azeban/config.hpp>
#include <azeban/forcing/no_forcing.hpp>
#include <azeban/grid.hpp>
#include <azeban/memory/workspace.hpp>
#include <azeban/operations/convolve.hpp>
#include <azeban/operations/fft.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/equations/incompressible_euler_cuda.hpp>
#endif
#include <azeban/profiler.hpp>
#include <type_traits>

namespace azeban {

template <int Dim, typename SpectralViscosity, typename Forcing = NoForcing>
class IncompressibleEuler final : public Equation<Dim> {
  using super = Equation<Dim>;
  static_assert(Dim == 2 || Dim == 3,
                "Incompressible Euler is only implemented for 2D and 3D");

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = Dim;

  template <bool enable = std::is_same_v<Forcing, NoForcing>,
            typename = std::enable_if_t<enable>>
  IncompressibleEuler(const Grid<dim_v> &grid,
                      const SpectralViscosity &visc,
                      zisa::device_type device,
                      bool has_tracer = false)
      : IncompressibleEuler(grid, visc, Forcing{}, device, has_tracer) {}
  IncompressibleEuler(const Grid<dim_v> &grid,
                      const SpectralViscosity &visc,
                      const Forcing &forcing,
                      zisa::device_type device,
                      bool has_tracer = false)
      : super(grid),
        device_(device),
        workspace_(),
        u_hat_({}, nullptr),
        u_({}, nullptr),
        B_hat_({}, nullptr),
        B_({}, nullptr),
        visc_(visc),
        forcing_(std::move(forcing)),
        has_tracer_(has_tracer) {
    const int n_vars_u = dim_v + (has_tracer ? 1 : 0);
    const int n_vars_B = (dim_v * dim_v + dim_v) / 2 + (has_tracer ? dim_v : 0);
    const auto shape_u_hat = grid.shape_fourier_pad(n_vars_u);
    const auto shape_u = grid.shape_phys_pad(n_vars_u);
    const auto shape_B_hat = grid.shape_fourier_pad(n_vars_B);
    const auto shape_B = grid.shape_phys_pad(n_vars_B);
    // Overlap buffer u_ with B_hat_ and u_hat_ with B_ in workspace
    const size_t size_u_hat = sizeof(complex_t) * zisa::product(shape_u_hat);
    const size_t size_u = sizeof(real_t) * zisa::product(shape_u);
    const size_t size_B_hat = sizeof(complex_t) * zisa::product(shape_B_hat);
    const size_t size_B = sizeof(real_t) * zisa::product(shape_B);
    const size_t size_u_and_B_hat = zisa::max(size_u, size_B_hat);
    const size_t size_u_hat_and_B = zisa::max(size_u_hat, size_B);
    const size_t u_and_B_hat_offset = 0;
    const size_t u_hat_and_B_offset
        = 256 * zisa::div_up(size_u_and_B_hat, size_t(256));
    const size_t size_workspace = u_hat_and_B_offset + size_u_hat_and_B;
    workspace_ = Workspace(size_workspace, device);
    u_hat_ = workspace_.get_view<complex_t>(u_hat_and_B_offset, shape_u_hat);
    u_ = workspace_.get_view<real_t>(u_and_B_hat_offset, shape_u);
    B_hat_ = workspace_.get_view<complex_t>(u_and_B_hat_offset, shape_B_hat);
    B_ = workspace_.get_view<real_t>(u_hat_and_B_offset, shape_B);
    fft_u_ = make_fft<dim_v>(u_hat_, u_, FFT_BACKWARD);
    fft_B_ = make_fft<dim_v>(B_hat_, B_, FFT_FORWARD);
  }
  IncompressibleEuler(const IncompressibleEuler &) = delete;
  IncompressibleEuler(IncompressibleEuler &&) = default;
  virtual ~IncompressibleEuler() = default;
  IncompressibleEuler &operator=(const IncompressibleEuler &) = delete;
  IncompressibleEuler &operator=(IncompressibleEuler &&) = default;

  virtual void
  dudt(const zisa::array_view<complex_t, dim_v + 1> &dudt_hat,
       const zisa::array_const_view<complex_t, dim_v + 1> &u_hat) override {
    LOG_ERR_IF(dudt_hat.shape(0) != u_hat_.shape(0),
               "Wrong number of variables");
    LOG_ERR_IF(u_hat.shape(0) != u_hat_.shape(0), "Wrong number of variables");
    AZEBAN_PROFILE_START("IncompressibleEuler::dudt");
    for (int i = 0; i < n_vars(); ++i) {
      copy_to_padded(component(u_hat_, i), component(u_hat, i));
    }
    fft_u_->backward();
    computeB();
    fft_B_->forward();
    computeDudt(dudt_hat, u_hat);
    AZEBAN_PROFILE_STOP("IncompressibleEuler::dudt");
  }

  virtual int n_vars() const override { return dim_v + (has_tracer_ ? 1 : 0); }

protected:
  using super::grid_;

private:
  zisa::device_type device_;
  Workspace workspace_;
  zisa::array_view<complex_t, dim_v + 1> u_hat_;
  zisa::array_view<real_t, dim_v + 1> u_;
  zisa::array_view<complex_t, dim_v + 1> B_hat_;
  zisa::array_view<real_t, dim_v + 1> B_;
  std::shared_ptr<FFT<dim_v>> fft_u_;
  std::shared_ptr<FFT<dim_v>> fft_B_;
  SpectralViscosity visc_;
  Forcing forcing_;
  bool has_tracer_;

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

  void computeB() {
    AZEBAN_PROFILE_START("IncompressibleEuler::computeB");
    if (device_ == zisa::device_type::cpu) {
      const real_t norm = 1.0
                          / (zisa::pow<dim_v>(grid_.N_phys)
                             * zisa::pow<dim_v>(grid_.N_phys_pad));
      if constexpr (dim_v == 2) {
        const unsigned stride = grid_.N_phys_pad * grid_.N_phys_pad;
#pragma omp parallel for collapse(2)
        for (zisa::int_t i = 0; i < u_.shape(1); ++i) {
          for (zisa::int_t j = 0; j < u_.shape(2); ++j) {
            const unsigned idx = i * grid_.N_phys_pad + j;
            const real_t u1 = u_(0, i, j);
            const real_t u2 = u_(1, i, j);
            incompressible_euler_2d_compute_B(
                stride, idx, norm, u1, u2, B_.raw());
            if (has_tracer_) {
              const real_t rho = u_(2, i, j);
              advection_2d_compute_B(
                  stride, idx, norm, rho, u1, u2, B_.raw() + 3 * stride);
            }
          }
        }
      } else {
        const unsigned stride
            = grid_.N_phys_pad * grid_.N_phys_pad * grid_.N_phys_pad;
#pragma omp parallel for collapse(3)
        for (zisa::int_t i = 0; i < u_.shape(1); ++i) {
          for (zisa::int_t j = 0; j < u_.shape(2); ++j) {
            for (zisa::int_t k = 0; k < u_.shape(3); ++k) {
              const unsigned idx = i * grid_.N_phys_pad * grid_.N_phys_pad
                                   + j * grid_.N_phys_pad + k;
              const real_t u1 = u_(0, i, j, k);
              const real_t u2 = u_(1, i, j, k);
              const real_t u3 = u_(2, i, j, k);
              incompressible_euler_3d_compute_B(
                  stride, idx, norm, u1, u2, u3, B_.raw());
              if (has_tracer_) {
                const real_t rho = u_(3, i, j, k);
                advection_3d_compute_B(
                    stride, idx, norm, rho, u1, u2, u3, B_.raw() + 6 * stride);
              }
            }
          }
        }
      }
    }
#if ZISA_HAS_CUDA
    else if (device_ == zisa::device_type::cuda) {
      if (has_tracer_) {
        incompressible_euler_compute_B_tracer_cuda<dim_v>(
            fft_B_->u(), fft_u_->u(), grid_);
      } else {
        incompressible_euler_compute_B_cuda<dim_v>(
            fft_B_->u(), fft_u_->u(), grid_);
      }
    }
#endif
    else {
      LOG_ERR("Unsupported memory location");
    }
    AZEBAN_PROFILE_STOP("IncompressibleEuler::computeB");
  }

  void
  computeDudt_cpu_2d(const zisa::array_view<complex_t, Dim + 1> &dudt_hat,
                     const zisa::array_const_view<complex_t, Dim + 1> &u_hat) {
    const unsigned stride_B = B_hat_.shape(1) * B_hat_.shape(2);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < zisa::integer_cast<int>(u_hat.shape(1)); ++i) {
      for (int j = 0; j < zisa::integer_cast<int>(u_hat.shape(2)); ++j) {
        const int i_B = i >= zisa::integer_cast<int>(u_hat.shape(1) / 2 + 1)
                            ? B_hat_.shape(1) - u_hat.shape(1) + i
                            : i;
        const unsigned idx_B = i_B * B_hat_.shape(2) + j;
        int i_ = i;
        if (i >= zisa::integer_cast<int>(u_hat.shape(1) / 2 + 1)) {
          i_ -= u_hat.shape(1);
        }
        const real_t k1 = 2 * zisa::pi * i_;
        const real_t k2 = 2 * zisa::pi * j;
        const real_t absk2 = k1 * k1 + k2 * k2;
        complex_t force1, force2;
        forcing_(0, k1, k2, &force1, &force2);
        complex_t L1_hat, L2_hat;
        // clang-format off
        incompressible_euler_2d_compute_L(
            k1, k2,
            absk2,
            stride_B, idx_B, B_hat_.raw(),
	    force1, force2,
            &L1_hat, &L2_hat
        );
        // clang-format on
        const real_t v = visc_.eval(zisa::sqrt(absk2));
        dudt_hat(0, i, j) = absk2 == 0 ? 0 : -L1_hat + v * u_hat(0, i, j);
        dudt_hat(1, i, j) = absk2 == 0 ? 0 : -L2_hat + v * u_hat(1, i, j);
        if (has_tracer_) {
          complex_t L3_hat;
          // clang-format off
          advection_2d(
              k1, k2,
              stride_B, idx_B, B_hat_.raw() + 3 * stride_B,
              &L3_hat
          );
          // clang-format on
          dudt_hat(2, i, j) = -L3_hat + v * u_hat(2, i, j);
        }
      }
    }
  }

  void
  computeDudt_cpu_3d(const zisa::array_view<complex_t, Dim + 1> &dudt_hat,
                     const zisa::array_const_view<complex_t, Dim + 1> &u_hat) {
    const unsigned stride_B
        = B_hat_.shape(1) * B_hat_.shape(2) * B_hat_.shape(3);
#pragma omp parallel for collapse(3)
    for (int i = 0; i < zisa::integer_cast<int>(u_hat.shape(1)); ++i) {
      for (int j = 0; j < zisa::integer_cast<int>(u_hat.shape(2)); ++j) {
        for (int k = 0; k < zisa::integer_cast<int>(u_hat.shape(3)); ++k) {
          const int i_B = i >= zisa::integer_cast<int>(u_hat.shape(1) / 2 + 1)
                              ? B_hat_.shape(1) - u_hat.shape(1) + i
                              : i;
          const int j_B = j >= zisa::integer_cast<int>(u_hat.shape(2) / 2 + 1)
                              ? B_hat_.shape(2) - u_hat.shape(2) + j
                              : j;
          const unsigned idx_B = i_B * B_hat_.shape(2) * B_hat_.shape(3)
                                 + j_B * B_hat_.shape(3) + k;
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
          const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;
          complex_t force1, force2, force3;
          forcing_(0, k1, k2, k3, &force1, &force2, &force3);
          complex_t L1_hat, L2_hat, L3_hat;
          // clang-format off
          incompressible_euler_3d_compute_L(
              k1, k2, k3,
              absk2,
              stride_B, idx_B, B_hat_.raw(),
	      force1, force2, force3,
              &L1_hat, &L2_hat, &L3_hat
          );

          const real_t v = visc_.eval(zisa::sqrt(absk2));
          dudt_hat(0, i, j, k) = absk2 == 0 ? 0 : -L1_hat + v * u_hat(0, i, j, k);
          dudt_hat(1, i, j, k) = absk2 == 0 ? 0 : -L2_hat + v * u_hat(1, i, j, k);
          dudt_hat(2, i, j, k) = absk2 == 0 ? 0 : -L3_hat + v * u_hat(2, i, j, k);
          if (has_tracer_) {
            complex_t L4_hat;
            advection_3d(
                k1, k2, k3,
                stride_B, idx_B, B_hat_.raw() + 6 * stride_B,
                &L4_hat
            );
            dudt_hat(3, i, j, k) = -L4_hat + v * u_hat(3, i, j, k);
          }
          // clang-format on
        }
      }
    }
  }

  void computeDudt(const zisa::array_view<complex_t, Dim + 1> &dudt_hat,
                   const zisa::array_const_view<complex_t, Dim + 1> &u_hat) {
    AZEBAN_PROFILE_START("IncompressibleEuler::computeDudt");
    if (device_ == zisa::device_type::cpu) {
      if constexpr (dim_v == 2) {
        computeDudt_cpu_2d(dudt_hat, u_hat);
      } else {
        computeDudt_cpu_3d(dudt_hat, u_hat);
      }
    }
#if ZISA_HAS_CUDA
    else if (device_ == zisa::device_type::cuda) {
      if (has_tracer_) {
        if constexpr (dim_v == 2) {
          incompressible_euler_2d_tracer_cuda(
              B_hat_, u_hat, dudt_hat, visc_, forcing_);
        } else {
          incompressible_euler_3d_tracer_cuda(
              B_hat_, u_hat, dudt_hat, visc_, forcing_);
        }
      } else {
        if constexpr (dim_v == 2) {
          incompressible_euler_2d_cuda(
              B_hat_, u_hat, dudt_hat, visc_, forcing_);
        } else {
          incompressible_euler_3d_cuda(
              B_hat_, u_hat, dudt_hat, visc_, forcing_);
        }
      }
    }
#endif
    else {
      LOG_ERR("Unsupported memory_location");
    }
    AZEBAN_PROFILE_STOP("IncompressibleEuler::computeDudt");
  }
};

}

#endif
