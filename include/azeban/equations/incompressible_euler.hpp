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
      LOG_ERR("Not yet implemented");
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
  zisa::int_t N_phys;
  zisa::int_t N_fourier;
  zisa::int_t N_phys_pad;
  zisa::int_t N_fourier_pad;
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
      LOG_ERR("Not yet implemented");
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
