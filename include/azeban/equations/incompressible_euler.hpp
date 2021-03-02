#ifndef INCOMPRESSIBLE_EULER_H_
#define INCOMPRESSIBLe_EULER_H_

#include "equation.hpp"
#include <azeban/config.hpp>
#include <azeban/fft.hpp>
#include <azeban/operations/convolve.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/equations/incompressible_euler_cuda.hpp>
#endif

namespace azeban {

template <int Dim, typename SpectralViscosity>
class IncompressibleEuler final : public Equation<complex_t, Dim + 1> {
  using super = Equation<complex_t, Dim + 1>;
  static_assert(Dim == 2 || Dim == 3,
                "Incompressible Euler is only implemented for 2D and 3D");
  static_assert(Dim == 2, "Only 2D Incompressible Euler supportet ATM");

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = Dim;

  IncompressibleEuler(zisa::int_t n,
                      const SpectralViscosity &visc,
                      zisa::device_type device = zisa::device_type::cpu)
      : device_(device), visc_(visc) {
    const zisa::int_t N_phys = n;
    const zisa::int_t N_fourier = N_phys / 2 + 1;
    const zisa::int_t N_phys_pad = 3. / 2 * N_phys + 1;
    const zisa::int_t N_fourier_pad = N_phys_pad / 2 + 1;
    zisa::shape_t<dim_v + 1> u_hat_shape;
    zisa::shape_t<dim_v + 1> u_shape;
    zisa::shape_t<dim_v + 1> B_hat_shape;
    zisa::shape_t<dim_v + 1> B_shape;
    u_hat_shape[0] = dim_v;
    u_shape[0] = dim_v;
    u_hat_shape[0] = dim_v * dim_v;
    u_shape[0] = dim_v * dim_v;
    for (int i = 0; i < dim_v - 1; ++i) {
      u_hat_shape[i + 1] = N_phys_pad;
      u_shape[i + 1] = N_phys_pad;
      B_hat_shape[i + 1] = N_phys_pad;
      B_shape[i + 1] = N_phys_pad;
    }
    u_hat_shape[dim_v] = N_fourier_pad;
    u_shape[dim_v] = N_phys_pad;
    B_hat_shape[dim_v] = N_fourier_pad;
    B_shape[dim_v] = N_phys_pad;
    u_hat_ = zisa::array<complex_t, dim_v + 1>(u_hat_shape, device);
    u_ = zisa::array<real_t, dim_v + 1>(u_shape, device);
    B_hat_ = zisa::array<complex_t, dim_v + 1>(u_hat_shape, device);
    B_ = zisa::array<real_t, dim_v + 1>(u_shape, device);
    fft_u_ = make_fft<dim_v>(zisa::array_view<complex_t, dim_v + 1>(u_hat_),
                             zisa::array_view<real_t, dim_v + 1>(u_));
    fft_B_ = make_fft<dim_v>(zisa::array_view<complex_t, dim_v + 1>(B_hat_),
                             zisa::array_view<real_t, dim_v + 1>(B_));
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
        incompressible_euler_2d_cuda(B_hat_, u_hat_, visc_);
      } else {
        incompressible_euler_3d_cuda(B_hat_, u_hat_, visc_);
      }
    }
#endif
    else {
      LOG_ERR("Unsupported memory_location");
    }
  }

private:
  zisa::device_type device_;
  zisa::array<complex_t, dim_v + 1> u_hat_;
  zisa::array<real_t, dim_v + 1> u_;
  // B is symmetric! Exploit to reduce memory usage => increase computational
  // intensity?
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
      incompressible_euler_compute_B_cuda<dim_v>(fft_B_->u(), fft_u_->u());
    }
#endif
    else {
      LOG_ERR("Unsupported memory location");
    }
  }
};

}

#endif
