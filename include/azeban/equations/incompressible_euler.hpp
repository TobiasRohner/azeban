#ifndef INCOMPRESSIBLE_EULER_H_
#define INCOMPRESSIBLE_EULER_H_

#include "advection_functions.hpp"
#include "equation.hpp"
#include "incompressible_euler_functions.hpp"
#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <azeban/operations/convolve.hpp>
#include <azeban/operations/fft.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/equations/incompressible_euler_cuda.hpp>
#endif
#include <azeban/profiler.hpp>

namespace azeban {

template <int Dim, typename SpectralViscosity>
class IncompressibleEuler final : public Equation<Dim> {
  using super = Equation<Dim>;
  static_assert(Dim == 2 || Dim == 3,
                "Incompressible Euler is only implemented for 2D and 3D");

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = Dim;

  IncompressibleEuler(const Grid<dim_v> &grid,
                      const SpectralViscosity &visc,
                      zisa::device_type device,
                      bool has_tracer = false)
      : super(grid), device_(device), visc_(visc), has_tracer_(has_tracer) {
    u_hat_ = grid.make_array_fourier_pad(dim_v + (has_tracer ? 1 : 0), device);
    u_ = grid.make_array_phys_pad(dim_v + (has_tracer ? 1 : 0), device);
    B_hat_ = grid.make_array_fourier_pad(
        (dim_v * dim_v + dim_v) / 2 + (has_tracer ? dim_v : 0), device);
    B_ = grid.make_array_phys_pad(
        (dim_v * dim_v + dim_v) / 2 + (has_tracer ? dim_v : 0), device);
    fft_u_ = make_fft<dim_v>(u_hat_, u_, FFT_BACKWARD);
    fft_B_ = make_fft<dim_v>(B_hat_, B_, FFT_FORWARD);
  }
  IncompressibleEuler(const IncompressibleEuler &) = delete;
  IncompressibleEuler(IncompressibleEuler &&) = default;
  virtual ~IncompressibleEuler() = default;
  IncompressibleEuler &operator=(const IncompressibleEuler &) = delete;
  IncompressibleEuler &operator=(IncompressibleEuler &&) = default;

  virtual void
  dudt(const zisa::array_view<complex_t, dim_v + 1> &u_hat) override {
    LOG_ERR_IF(u_hat.shape(0) != u_hat_.shape(0), "Wrong number of variables");
    AZEBAN_PROFILE_START("IncompressibleEuler::dudt");
    for (int i = 0; i < n_vars(); ++i) {
      copy_to_padded(
          component(u_hat_, i),
          zisa::array_const_view<complex_t, dim_v>(component(u_hat, i)),
          complex_t(0));
    }
    fft_u_->backward();
    computeB();
    fft_B_->forward();
    computeDudt(u_hat);
    AZEBAN_PROFILE_STOP("IncompressibleEuler::dudt");
  }

  virtual int n_vars() const override { return dim_v + (has_tracer_ ? 1 : 0); }

protected:
  using super::grid_;

private:
  zisa::device_type device_;
  zisa::array<complex_t, dim_v + 1> u_hat_;
  zisa::array<real_t, dim_v + 1> u_;
  zisa::array<complex_t, dim_v + 1> B_hat_;
  zisa::array<real_t, dim_v + 1> B_;
  std::shared_ptr<FFT<dim_v>> fft_u_;
  std::shared_ptr<FFT<dim_v>> fft_B_;
  SpectralViscosity visc_;
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
    AZEBAN_PROFILE_START("IncompressibleEuler::computeB");
    if (device_ == zisa::device_type::cpu) {
      const real_t norm = 1.0
                          / (zisa::pow<dim_v>(grid_.N_phys)
                             * zisa::pow<dim_v>(grid_.N_phys_pad));
      if constexpr (dim_v == 2) {
        const unsigned stride = grid_.N_phys_pad * grid_.N_phys_pad;
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

  void computeDudt(const zisa::array_view<complex_t, Dim + 1> &u_hat) {
    AZEBAN_PROFILE_START("IncompressibleEuler::computeDudt");
    if (device_ == zisa::device_type::cpu) {
      if constexpr (dim_v == 2) {
        const unsigned stride_B = B_hat_.shape(1) * B_hat_.shape(2);
        for (int i = 0; i < zisa::integer_cast<int>(u_hat.shape(1)); ++i) {
          const int i_B = i >= zisa::integer_cast<int>(u_hat.shape(1) / 2 + 1)
                              ? B_hat_.shape(1) - u_hat.shape(1) + i
                              : i;
          for (int j = 0; j < zisa::integer_cast<int>(u_hat.shape(2)); ++j) {
            const unsigned idx_B = i_B * B_hat_.shape(2) + j;
            int i_ = i;
            if (i >= zisa::integer_cast<int>(u_hat.shape(1) / 2 + 1)) {
              i_ -= u_hat.shape(1);
            }
            const real_t k1 = 2 * zisa::pi * i_;
            const real_t k2 = 2 * zisa::pi * j;
            const real_t absk2 = k1 * k1 + k2 * k2;
            complex_t L1_hat, L2_hat;
            incompressible_euler_2d_compute_L(
                k1, k2, absk2, stride_B, idx_B, B_hat_.raw(), &L1_hat, &L2_hat);
            const real_t v = visc_.eval(zisa::sqrt(absk2));
            u_hat(0, i, j) = absk2 == 0 ? 0 : -L1_hat + v * u_hat(0, i, j);
            u_hat(1, i, j) = absk2 == 0 ? 0 : -L2_hat + v * u_hat(1, i, j);
            if (has_tracer_) {
              complex_t L3_hat;
              advection_2d(k1,
                           k2,
                           stride_B,
                           idx_B,
                           B_hat_.raw() + 3 * stride_B,
                           &L3_hat);
              u_hat(2, i, j) = -L3_hat + v * u_hat(2, i, j);
            }
          }
        }
      } else {
        const unsigned stride_B
            = B_hat_.shape(1) * B_hat_.shape(2) * B_hat_.shape(3);
        for (int i = 0; i < zisa::integer_cast<int>(u_hat.shape(1)); ++i) {
          const int i_B = i >= zisa::integer_cast<int>(u_hat.shape(1) / 2 + 1)
                              ? B_hat_.shape(1) - u_hat.shape(1) + i
                              : i;
          for (int j = 0; j < zisa::integer_cast<int>(u_hat.shape(2)); ++j) {
            const int j_B = j >= zisa::integer_cast<int>(u_hat.shape(2) / 2 + 1)
                                ? B_hat_.shape(2) - u_hat.shape(2) + j
                                : j;
            for (int k = 0; k < zisa::integer_cast<int>(u_hat.shape(3)); ++k) {
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
              complex_t L1_hat, L2_hat, L3_hat;
              incompressible_euler_3d_compute_L(k1,
                                                k2,
                                                k3,
                                                absk2,
                                                stride_B,
                                                idx_B,
                                                B_hat_.raw(),
                                                &L1_hat,
                                                &L2_hat,
                                                &L3_hat);
              const real_t v = visc_.eval(zisa::sqrt(absk2));
              u_hat(0, i, j, k)
                  = absk2 == 0 ? 0 : -L1_hat + v * u_hat(0, i, j, k);
              u_hat(1, i, j, k)
                  = absk2 == 0 ? 0 : -L2_hat + v * u_hat(1, i, j, k);
              u_hat(2, i, j, k)
                  = absk2 == 0 ? 0 : -L3_hat + v * u_hat(2, i, j, k);
              if (has_tracer_) {
                complex_t L4_hat;
                advection_3d(k1,
                             k2,
                             k3,
                             stride_B,
                             idx_B,
                             B_hat_.raw() + 6 * stride_B,
                             &L4_hat);
                u_hat(4, i, j, k) = -L4_hat + v * u_hat(3, i, j, k);
              }
            }
          }
        }
      }
    }
#if ZISA_HAS_CUDA
    else if (device_ == zisa::device_type::cuda) {
      if (has_tracer_) {
        if constexpr (dim_v == 2) {
          incompressible_euler_2d_tracer_cuda(B_hat_, u_hat, visc_);
        } else {
          incompressible_euler_3d_tracer_cuda(B_hat_, u_hat, visc_);
        }
      } else {
        if constexpr (dim_v == 2) {
          incompressible_euler_2d_cuda(B_hat_, u_hat, visc_);
        } else {
          incompressible_euler_3d_cuda(B_hat_, u_hat, visc_);
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
