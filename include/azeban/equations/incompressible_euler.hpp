#ifndef INCOMPRESSIBLE_EULER_H_
#define INCOMPRESSIBLE_EULER_H_

#include "equation.hpp"
#include <azeban/config.hpp>
#include <azeban/fft.hpp>
#include <azeban/grid.hpp>
#include <azeban/operations/convolve.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/equations/incompressible_euler_cuda.hpp>
#endif

namespace azeban {

template <int Dim, typename SpectralViscosity>
class IncompressibleEuler final : public Equation<complex_t, Dim> {
  using super = Equation<complex_t, Dim>;
  static_assert(Dim == 2 || Dim == 3,
                "Incompressible Euler is only implemented for 2D and 3D");

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = Dim;

  IncompressibleEuler(const Grid<dim_v> &grid,
                      const SpectralViscosity &visc,
                      zisa::device_type device)
      : super(grid), device_(device), visc_(visc) {
    u_hat_ = grid.make_array_fourier_pad(dim_v, device);
    u_ = grid.make_array_phys_pad(dim_v, device);
    B_hat_ = grid.make_array_fourier_pad((dim_v * dim_v + dim_v) / 2, device);
    B_ = grid.make_array_phys_pad((dim_v * dim_v + dim_v) / 2, device);
    fft_u_ = make_fft<dim_v>(u_hat_, u_);
    fft_B_ = make_fft<dim_v>(B_hat_, B_);
  }
  IncompressibleEuler(const IncompressibleEuler &) = delete;
  IncompressibleEuler(IncompressibleEuler &&) = default;
  virtual ~IncompressibleEuler() = default;
  IncompressibleEuler &operator=(const IncompressibleEuler &) = delete;
  IncompressibleEuler &operator=(IncompressibleEuler &&) = default;

  virtual void
  dudt(const zisa::array_view<complex_t, dim_v + 1> &u_hat) override {
    for (int i = 0; i < dim_v; ++i) {
      copy_to_padded(
          component(u_hat_, i),
          zisa::array_const_view<complex_t, dim_v>(component(u_hat, i)),
          complex_t(0));
    }
    fft_u_->backward();
    computeB();
    fft_B_->forward();
    if (device_ == zisa::device_type::cpu) {
      if constexpr (dim_v == 2) {
        for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
          const int i_B = i >= u_hat.shape(1) / 2 + 1
                              ? B_hat_.shape(1) - u_hat.shape(1) + i
                              : i;
          for (zisa::int_t j = 0; j < u_hat.shape(2); ++j) {
            int i_ = i;
            if (i >= u_hat.shape(1) / 2 + 1) {
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
            u_hat(0, i, j) = absk2 == 0 ? 0 : -L1_hat + v * u_hat(0, i, j);
            u_hat(1, i, j) = absk2 == 0 ? 0 : -L2_hat + v * u_hat(1, i, j);
          }
        }
      } else {
        LOG_ERR("Not yet implemented");
      }
    }
#if ZISA_HAS_CUDA
    else if (device_ == zisa::device_type::cuda) {
      if constexpr (dim_v == 2) {
        incompressible_euler_2d_cuda(B_hat_, u_hat, visc_);
      } else {
        incompressible_euler_3d_cuda(B_hat_, u_hat, visc_);
      }
    }
#endif
    else {
      LOG_ERR("Unsupported memory_location");
    }
  }

  using super::grid;
  virtual int n_vars() const override { return dim_v; }

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
      incompressible_euler_compute_B_cuda<dim_v>(
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
