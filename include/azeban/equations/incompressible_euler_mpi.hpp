#ifndef INCOMPRESSIBLE_EULER_MPI_H_
#define INCOMPRESSIBLE_EULER_MPI_H_

#include "advection_functions.hpp"
#include "equation.hpp"
#include "incompressible_euler_functions.hpp"
#include <azeban/config.hpp>
#include <azeban/cuda/equations/incompressible_euler_cuda.hpp>
#include <azeban/operations/fft_mpi_factory.hpp>
#include <azeban/profiler.hpp>

namespace azeban {

template <int Dim, typename SpectralViscosity>
class IncompressibleEuler_MPI final : public Equation<Dim> {
  using super = Equation<Dim>;
  static_assert(Dim == 2 || Dim == 3,
                "Incompressible Euler is only implemented for 2D and 3D");

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = Dim;

  IncompressibleEuler_MPI(const Grid<dim_v> &grid,
                          MPI_Comm comm,
                          const SpectralViscosity &visc,
                          bool has_tracer = false)
      : super(grid), comm_(comm), visc_(visc), has_tracer_(has_tracer) {
    // TODO: Actually pad the padded arrays
    grid_.N_phys_pad = grid_.N_phys;
    grid_.N_fourier_pad = grid_.N_fourier;
    const zisa::int_t n_var_u = dim_v + (has_tracer ? 1 : 0);
    const zisa::int_t n_var_B
        = (dim_v * dim_v + dim_v) / 2 + (has_tracer ? dim_v : 0);
    h_u_hat_pad_
        = grid_.make_array_fourier_pad(n_var_u, zisa::device_type::cpu, comm);
    d_u_hat_pad_
        = grid_.make_array_fourier_pad(n_var_u, zisa::device_type::cuda, comm);
    u_pad_ = grid_.make_array_phys_pad(n_var_u, zisa::device_type::cuda, comm);
    B_pad_ = grid_.make_array_phys_pad(n_var_B, zisa::device_type::cuda, comm);
    d_B_hat_pad_
        = grid_.make_array_fourier_pad(n_var_B, zisa::device_type::cuda, comm);
    h_B_hat_pad_
        = grid_.make_array_fourier_pad(n_var_B, zisa::device_type::cpu, comm);
    B_hat_ = grid_.make_array_fourier(n_var_B, zisa::device_type::cpu, comm);
    fft_u_ = make_fft_mpi<dim_v>(d_u_hat_pad_, u_pad_, comm, FFT_BACKWARD);
    fft_B_ = make_fft_mpi<dim_v>(d_B_hat_pad_, B_pad_, comm, FFT_FORWARD);
  }
  IncompressibleEuler_MPI(const IncompressibleEuler_MPI &) = delete;
  IncompressibleEuler_MPI(IncompressibleEuler_MPI &&) = default;
  virtual ~IncompressibleEuler_MPI() = default;
  IncompressibleEuler_MPI &operator=(const IncompressibleEuler_MPI &) = delete;
  IncompressibleEuler_MPI &operator=(IncompressibleEuler_MPI &&) = default;

  virtual void
  dudt(const zisa::array_view<complex_t, dim_v + 1> &u_hat) override {
    LOG_ERR_IF(u_hat.memory_location() != zisa::device_type::cpu,
               "Euler MPI needs CPU arrays");
    LOG_ERR_IF(u_hat.shape(0) != h_u_hat_pad_.shape(0),
               "Wrong number of variables");
    AZEBAN_PROFILE_START("IncompressibleEuler_MPI::dudt");
    const zisa::int_t n_var_u = dim_v + (has_tracer_ ? 1 : 0);
    const zisa::int_t n_var_B
        = (dim_v * dim_v + dim_v) / 2 + (has_tracer_ ? dim_v : 0);
    for (zisa::int_t i = 0; i < n_var_u; ++i) {
      // TODO: Change this to a padded copy
      zisa::copy(component(h_u_hat_pad_, i), component(u_hat, i));
    }
    zisa::copy(d_u_hat_pad_, h_u_hat_pad_);
    fft_u_->backward();
    computeB();
    fft_B_->forward();
    zisa::copy(h_B_hat_pad_, d_B_hat_pad_);
    for (zisa::int_t i = 0; i < n_var_B; ++i) {
      // TODO: Change this to a padded copy
      zisa::copy(component(B_hat_, i), component(h_B_hat_pad_, i));
    }
    computeDudt(u_hat);
    AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::dudt");
  }

  virtual int n_vars() const override { return dim_v + (has_tracer_ ? 1 : 0); }

protected:
  using super::grid_;

private:
  MPI_Comm comm_;
  SpectralViscosity visc_;
  zisa::array<complex_t, dim_v + 1> h_u_hat_pad_;
  zisa::array<complex_t, dim_v + 1> d_u_hat_pad_;
  zisa::array<real_t, dim_v + 1> u_pad_;
  zisa::array<real_t, dim_v + 1> B_pad_;
  zisa::array<complex_t, dim_v + 1> d_B_hat_pad_;
  zisa::array<complex_t, dim_v + 1> h_B_hat_pad_;
  zisa::array<complex_t, dim_v + 1> B_hat_;
  std::shared_ptr<FFT<dim_v>> fft_u_;
  std::shared_ptr<FFT<dim_v>> fft_B_;
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
    AZEBAN_PROFILE_START("IncompressibleEuler_MPI::computeB");
    if (has_tracer_) {
      incompressible_euler_compute_B_tracer_cuda<dim_v>(
          fft_B_->u(), fft_u_->u(), grid_);
    } else {
      incompressible_euler_compute_B_cuda<dim_v>(
          fft_B_->u(), fft_u_->u(), grid_);
    }
    AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::computeB");
  }

  void computeDudt(const zisa::array_view<complex_t, dim_v + 1> &u_hat) {
    AZEBAN_PROFILE_START("IncompressibleEuler_MPI::computeDudt");
    if constexpr (dim_v == 2) {
      const zisa::int_t i_base = grid_.i_fourier(0, comm_);
      const zisa::int_t j_base = grid_.j_fourier(0, comm_);
      const auto shape_phys = grid_.shape_phys(1);
      const unsigned stride_B = B_hat_.shape(1) * B_hat_.shape(2);
      for (int i = 0; i < zisa::integer_cast<int>(u_hat.shape(1)); ++i) {
        for (int j = 0; j < zisa::integer_cast<int>(u_hat.shape(2)); ++j) {
          const unsigned idx_B = i * B_hat_.shape(2) + j;
          int i_ = i_base + i;
          if (i_ >= zisa::integer_cast<int>(shape_phys[1] / 2 + 1)) {
            i_ -= shape_phys[1];
          }
          int j_ = j_base + j;
          if (j_ >= zisa::integer_cast<int>(shape_phys[2] / 2 + 1)) {
            j_ -= shape_phys[2];
          }
          const real_t k1 = 2 * zisa::pi * i_;
          const real_t k2 = 2 * zisa::pi * j_;
          const real_t absk2 = k1 * k1 + k2 * k2;
          complex_t L1_hat, L2_hat;
          incompressible_euler_2d_compute_L(
              k1, k2, absk2, stride_B, idx_B, B_hat_.raw(), &L1_hat, &L2_hat);
          const real_t v = visc_.eval(zisa::sqrt(absk2));
          u_hat(0, i, j) = absk2 == 0 ? 0 : -L1_hat + v * u_hat(0, i, j);
          u_hat(1, i, j) = absk2 == 0 ? 0 : -L2_hat + v * u_hat(1, i, j);
          if (has_tracer_) {
            complex_t L3_hat;
            advection_2d(
                k1, k2, stride_B, idx_B, B_hat_.raw() + 3 * stride_B, &L3_hat);
            u_hat(2, i, j) = -L3_hat + v * u_hat(2, i, j);
          }
        }
      }
    } else {
      const zisa::int_t i_base = grid_.i_fourier(0, comm_);
      const zisa::int_t j_base = grid_.j_fourier(0, comm_);
      const zisa::int_t k_base = grid_.k_fourier(0, comm_);
      const auto shape_phys = grid_.shape_phys(1);
      const unsigned stride_B
          = B_hat_.shape(1) * B_hat_.shape(2) * B_hat_.shape(3);
      for (int i = 0; i < zisa::integer_cast<int>(u_hat.shape(1)); ++i) {
        for (int j = 0; j < zisa::integer_cast<int>(u_hat.shape(2)); ++j) {
          for (int k = 0; k < zisa::integer_cast<int>(u_hat.shape(3)); ++k) {
            const unsigned idx_B = i * B_hat_.shape(2) * B_hat_.shape(3)
                                   + j * B_hat_.shape(3) + k;
            int i_ = i_base + i;
            int j_ = j_base + j;
            int k_ = k_base + k;
            if (i_ >= zisa::integer_cast<int>(shape_phys[1] / 2 + 1)) {
              i_ -= shape_phys[1];
            }
            if (j_ >= zisa::integer_cast<int>(shape_phys[2] / 2 + 1)) {
              j_ -= shape_phys[2];
            }
            if (k_ >= zisa::integer_cast<int>(shape_phys[3] / 2 + 1)) {
              k_ -= shape_phys[3];
            }
            const real_t k1 = 2 * zisa::pi * i_;
            const real_t k2 = 2 * zisa::pi * j_;
            const real_t k3 = 2 * zisa::pi * k_;
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
              u_hat(3, i, j, k) = -L4_hat + v * u_hat(3, i, j, k);
            }
          }
        }
      }
    }
    AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::computeDudt");
  }
};

}

#endif
