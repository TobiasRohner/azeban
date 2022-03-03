#ifndef AZEBAN_EQUATIONS_INCOMPRESSIBLE_EULER_MPI_HPP_
#define AZEBAN_EQUATIONS_INCOMPRESSIBLE_EULER_MPI_HPP_

#include <azeban/equations/advection_functions.hpp>
#include <azeban/equations/equation.hpp>
#include <azeban/equations/incompressible_euler_functions.hpp>
#include <azeban/forcing/no_forcing.hpp>
#include <azeban/memory/workspace.hpp>
#include <azeban/operations/fft.hpp>
#include <azeban/operations/transpose.hpp>
#include <azeban/profiler.hpp>
#include <mpi.h>
#if ZISA_HAS_CUDA
#include <azeban/cuda/equations/incompressible_euler_mpi_cuda.hpp>
#endif

namespace azeban {

template <int Dim>
class IncompressibleEuler_MPI_Base : public Equation<Dim> {
  using super = Equation<Dim>;
  static_assert(Dim == 2 || Dim == 3,
                "Incompressible Euler is only implemente for 2D and 3D");

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = Dim;

  IncompressibleEuler_MPI_Base(const Grid<dim_v> &grid,
                               MPI_Comm comm,
                               zisa::device_type device,
                               bool has_tracer = false);
  IncompressibleEuler_MPI_Base(const IncompressibleEuler_MPI_Base &) = delete;
  IncompressibleEuler_MPI_Base(IncompressibleEuler_MPI_Base &&) = default;
  virtual ~IncompressibleEuler_MPI_Base() = default;
  IncompressibleEuler_MPI_Base &operator=(const IncompressibleEuler_MPI_Base &)
      = delete;
  IncompressibleEuler_MPI_Base &operator=(IncompressibleEuler_MPI_Base &&)
      = default;

  virtual int n_vars() const override { return dim_v + (has_tracer_ ? 1 : 0); }

  virtual void *get_fft_work_area() const override;

protected:
  using super::grid_;
  MPI_Comm comm_;
  zisa::device_type device_;
  bool has_tracer_;
  zisa::array_view<complex_t, dim_v + 1> B_hat_;

  static zisa::array_view<complex_t, dim_v>
  component(const zisa::array_view<complex_t, dim_v + 1> &arr, int dim);
  static zisa::array_const_view<complex_t, dim_v>
  component(const zisa::array_const_view<complex_t, dim_v + 1> &arr, int dim);
  static zisa::array_view<complex_t, dim_v>
  component(zisa::array<complex_t, dim_v + 1> &arr, int dim);
  static zisa::array_const_view<complex_t, dim_v>
  component(const zisa::array<complex_t, dim_v + 1> &arr, int dim);

  void computeBhat(const zisa::array_const_view<complex_t, dim_v + 1> &u_hat);

private:
  int mpi_size_;
  int mpi_rank_;
  Workspace ws1_;
  Workspace ws2_;
  Workspace ws_fft_;
  std::shared_ptr<FFT<dim_v, complex_t>> fft_u_yz_;
  std::shared_ptr<FFT<dim_v, real_t>> fft_u_x_;
  std::shared_ptr<FFT<dim_v, complex_t>> fft_B_yz_;
  std::shared_ptr<FFT<dim_v, real_t>> fft_B_x_;
  zisa::array_view<complex_t, dim_v + 1> u_hat_pad_;
  zisa::array_view<complex_t, dim_v + 1> u_yz_;
  zisa::array_view<complex_t, dim_v + 1> u_yz_trans_;
  zisa::array_view<complex_t, dim_v + 1> u_yz_trans_pad_;
  zisa::array_view<real_t, dim_v + 1> u_xyz_trans_;
  zisa::array_view<real_t, dim_v + 1> B_xyz_trans_;
  zisa::array_view<complex_t, dim_v + 1> B_yz_trans_pad_;
  zisa::array_view<complex_t, dim_v + 1> B_yz_trans_;
  zisa::array_view<complex_t, dim_v + 1> B_yz_;
  zisa::array_view<complex_t, dim_v + 1> B_hat_pad_;
  std::shared_ptr<Transpose<dim_v>> transpose_u_;
  std::shared_ptr<Transpose<dim_v>> transpose_B_;
  Workspace ws1_trans_;
  Workspace ws2_trans_;
  zisa::array_view<complex_t, dim_v + 2> trans_u_sendbuf_;
  zisa::array_view<complex_t, dim_v + 2> trans_u_recvbuf_;
  zisa::array_view<complex_t, dim_v + 2> trans_B_sendbuf_;
  zisa::array_view<complex_t, dim_v + 2> trans_B_recvbuf_;

  void
  compute_u_hat_pad(const zisa::array_const_view<complex_t, dim_v + 1> &u_hat);
  void compute_u_yz();
  void compute_u_yz_trans();
  void compute_u_yz_trans_pad();
  void compute_u_xyz_trans();
  void compute_B_xyz_trans();
  void compute_B_xyz_trans_cpu();
  void compute_B_yz_trans_pad();
  void compute_B_yz_trans();
  void compute_B_yz();
  void compute_B_hat_pad();
  void compute_B_hat();
};

template <int Dim, typename SpectralViscosity, typename Forcing = NoForcing>
class IncompressibleEuler_MPI {
  static_assert(Dim == 2 || Dim == 3,
                "Incompressible Euler is only implemented for 2D and 3D");
};

template <typename SpectralViscosity, typename Forcing>
class IncompressibleEuler_MPI<2, SpectralViscosity, Forcing>
    : public IncompressibleEuler_MPI_Base<2> {
  using super = IncompressibleEuler_MPI_Base<2>;

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = 2;

  template <bool enable = std::is_same_v<Forcing, NoForcing>,
            typename = std::enable_if_t<enable>>
  IncompressibleEuler_MPI(const Grid<2> &grid,
                          MPI_Comm comm,
                          const SpectralViscosity &visc,
                          zisa::device_type device,
                          bool has_tracer = false)
      : IncompressibleEuler_MPI(
          grid, comm, visc, NoForcing{}, device, has_tracer) {}
  IncompressibleEuler_MPI(const Grid<2> &grid,
                          MPI_Comm comm,
                          const SpectralViscosity &visc,
                          const Forcing &forcing,
                          zisa::device_type device,
                          bool has_tracer = false)
      : super(grid, comm, device, has_tracer), visc_(visc), forcing_(forcing) {}
  IncompressibleEuler_MPI(const IncompressibleEuler_MPI &) = delete;
  IncompressibleEuler_MPI(IncompressibleEuler_MPI &&) = default;
  virtual ~IncompressibleEuler_MPI() = default;
  IncompressibleEuler_MPI &operator=(const IncompressibleEuler_MPI &) = delete;
  IncompressibleEuler_MPI &operator=(IncompressibleEuler_MPI &&) = default;

  virtual void
  dudt(const zisa::array_view<complex_t, 3> &dudt_hat,
       const zisa::array_const_view<complex_t, 3> &u_hat) override {
    AZEBAN_PROFILE_START("IncompressibleEuler_MPI::dudt");
    computeBhat(u_hat);
    computeDudt(dudt_hat, u_hat);
    AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::dudt");
  }

  using super::n_vars;

protected:
  using super::B_hat_;
  using super::device_;
  using super::grid_;

  using super::computeBhat;

private:
  SpectralViscosity visc_;
  Forcing forcing_;

  void computeDudt(const zisa::array_view<complex_t, 3> &dudt_hat,
                   const zisa::array_const_view<complex_t, 3> &u_hat) {
    AZEBAN_PROFILE_START("IncompressibleEuler_MPI::computeDudt");
    if (device_ == zisa::device_type::cpu) {
      computeDudt_cpu(dudt_hat, u_hat);
    }
#if ZISA_HAS_CUDA
    else if (device_ == zisa::device_type::cuda) {
      computeDudt_cuda(dudt_hat, u_hat);
    }
#endif
    else {
      LOG_ERR("Unsupported device");
    }
    AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::computeDudt");
  }

  void computeDudt_cpu(const zisa::array_view<complex_t, 3> &dudt_hat,
                       const zisa::array_const_view<complex_t, 3> &u_hat) {
    const zisa::int_t i_base = grid_.i_fourier(0, comm_);
    const zisa::int_t j_base = grid_.j_fourier(0, comm_);
    const auto shape_phys = grid_.shape_phys(1);
    const unsigned long stride_B = B_hat_.shape(1) * B_hat_.shape(2);
    const long nx = zisa::integer_cast<long>(u_hat.shape(1));
    const long ny = zisa::integer_cast<long>(u_hat.shape(2));
#pragma omp parallel for collapse(2)
    for (long i = 0; i < nx; ++i) {
      for (long j = 0; j < ny; ++j) {
        const unsigned long idx_B = i * B_hat_.shape(2) + j;
        long i_ = i_base + i;
        if (i_ >= zisa::integer_cast<long>(shape_phys[1] / 2 + 1)) {
          i_ -= shape_phys[1];
        }
        long j_ = j_base + j;
        if (j_ >= zisa::integer_cast<long>(shape_phys[2] / 2 + 1)) {
          j_ -= shape_phys[2];
        }
        const real_t k1 = 2 * zisa::pi * i_;
        const real_t k2 = 2 * zisa::pi * j_;
        const real_t absk2 = k1 * k1 + k2 * k2;
        complex_t force1, force2;
        forcing_(0, k2, k1, &force1, &force2);
        complex_t L1_hat, L2_hat;
        incompressible_euler_2d_compute_L(k2,
                                          k1,
                                          absk2,
                                          stride_B,
                                          idx_B,
                                          B_hat_.raw(),
                                          force1,
                                          force2,
                                          &L1_hat,
                                          &L2_hat);
        const real_t v = visc_.eval(zisa::sqrt(absk2));
        dudt_hat(0, i, j) = absk2 == 0 ? 0 : -L1_hat + v * u_hat(0, i, j);
        dudt_hat(1, i, j) = absk2 == 0 ? 0 : -L2_hat + v * u_hat(1, i, j);
        if (has_tracer_) {
          complex_t L3_hat;
          advection_2d(
              k2, k1, stride_B, idx_B, B_hat_.raw() + 3 * stride_B, &L3_hat);
          dudt_hat(2, i, j) = -L3_hat + v * u_hat(2, i, j);
        }
      }
    }
  }

#if ZISA_HAS_CUDA
  void computeDudt_cuda(const zisa::array_view<complex_t, 3> &dudt_hat,
                        const zisa::array_const_view<complex_t, 3> &u_hat) {
    AZEBAN_PROFILE_START("IncompressibleEuler_MPI::computeDudt");
    const zisa::int_t i_base = grid_.i_fourier(0, comm_);
    const zisa::int_t j_base = grid_.j_fourier(0, comm_);
    const auto shape_phys = grid_.shape_phys(1);
    if (has_tracer_) {
      incompressible_euler_mpi_2d_tracer_cuda(
          B_hat_, u_hat, dudt_hat, visc_, forcing_, i_base, j_base, shape_phys);
    } else {
      incompressible_euler_mpi_2d_cuda(
          B_hat_, u_hat, dudt_hat, visc_, forcing_, i_base, j_base, shape_phys);
    }
    AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::computeDudt");
  }
#endif
};

template <typename SpectralViscosity, typename Forcing>
class IncompressibleEuler_MPI<3, SpectralViscosity, Forcing>
    : public IncompressibleEuler_MPI_Base<3> {
  using super = IncompressibleEuler_MPI_Base<3>;

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = 3;

  template <bool enable = std::is_same_v<Forcing, NoForcing>,
            typename = std::enable_if_t<enable>>
  IncompressibleEuler_MPI(const Grid<3> &grid,
                          MPI_Comm comm,
                          const SpectralViscosity &visc,
                          zisa::device_type device,
                          bool has_tracer = false)
      : IncompressibleEuler_MPI(
          grid, comm, visc, NoForcing{}, device, has_tracer) {}
  IncompressibleEuler_MPI(const Grid<3> &grid,
                          MPI_Comm comm,
                          const SpectralViscosity &visc,
                          const Forcing &forcing,
                          zisa::device_type device,
                          bool has_tracer = false)
      : super(grid, comm, device, has_tracer), visc_(visc), forcing_(forcing) {}
  IncompressibleEuler_MPI(const IncompressibleEuler_MPI &) = delete;
  IncompressibleEuler_MPI(IncompressibleEuler_MPI &&) = default;
  virtual ~IncompressibleEuler_MPI() = default;
  IncompressibleEuler_MPI &operator=(const IncompressibleEuler_MPI &) = delete;
  IncompressibleEuler_MPI &operator=(IncompressibleEuler_MPI &&) = default;

  virtual void
  dudt(const zisa::array_view<complex_t, 4> &dudt_hat,
       const zisa::array_const_view<complex_t, 4> &u_hat) override {
    AZEBAN_PROFILE_START("IncompressibleEuler_MPI::dudt");
    computeBhat(u_hat);
    computeDudt(dudt_hat, u_hat);
    AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::dudt");
  }

  using super::n_vars;

protected:
  using super::B_hat_;
  using super::device_;
  using super::grid_;

  using super::computeBhat;

private:
  SpectralViscosity visc_;
  Forcing forcing_;

  void computeDudt(const zisa::array_view<complex_t, 4> &dudt_hat,
                   const zisa::array_const_view<complex_t, 4> &u_hat) {
    AZEBAN_PROFILE_START("IncompressibleEuler_MPI::computeDudt");
    if (device_ == zisa::device_type::cpu) {
      computeDudt_cpu(dudt_hat, u_hat);
    }
#if ZISA_HAS_CUDA
    else if (device_ == zisa::device_type::cuda) {
      computeDudt_cuda(dudt_hat, u_hat);
    }
#endif
    else {
      LOG_ERR("Unsupported device");
    }
    AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::computeDudt");
  }

  void computeDudt_cpu(const zisa::array_view<complex_t, 4> &dudt_hat,
                       const zisa::array_const_view<complex_t, 4> &u_hat) {
    const zisa::int_t i_base = grid_.i_fourier(0, comm_);
    const zisa::int_t j_base = grid_.j_fourier(0, comm_);
    const zisa::int_t k_base = grid_.k_fourier(0, comm_);
    const auto shape_phys = grid_.shape_phys(1);
    const unsigned long stride_B
        = B_hat_.shape(1) * B_hat_.shape(2) * B_hat_.shape(3);
    const long nx = zisa::integer_cast<long>(u_hat.shape(1));
    const long ny = zisa::integer_cast<long>(u_hat.shape(2));
    const long nz = zisa::integer_cast<long>(u_hat.shape(3));
#pragma omp parallel for collapse(3)
    for (long i = 0; i < nx; ++i) {
      for (long j = 0; j < ny; ++j) {
        for (long k = 0; k < nz; ++k) {
          const unsigned long idx_B
              = i * B_hat_.shape(2) * B_hat_.shape(3) + j * B_hat_.shape(3) + k;
          long i_ = i_base + i;
          long j_ = j_base + j;
          long k_ = k_base + k;
          if (i_ >= zisa::integer_cast<long>(shape_phys[1] / 2 + 1)) {
            i_ -= shape_phys[1];
          }
          if (j_ >= zisa::integer_cast<long>(shape_phys[2] / 2 + 1)) {
            j_ -= shape_phys[2];
          }
          if (k_ >= zisa::integer_cast<long>(shape_phys[3] / 2 + 1)) {
            k_ -= shape_phys[3];
          }
          const real_t k1 = 2 * zisa::pi * i_;
          const real_t k2 = 2 * zisa::pi * j_;
          const real_t k3 = 2 * zisa::pi * k_;
          const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;
          complex_t force1, force2, force3;
          forcing_(0, k3, k2, k1, &force1, &force2, &force3);
          complex_t L1_hat, L2_hat, L3_hat;
          incompressible_euler_3d_compute_L(k3,
                                            k2,
                                            k1,
                                            absk2,
                                            stride_B,
                                            idx_B,
                                            B_hat_.raw(),
                                            force1,
                                            force2,
                                            force3,
                                            &L1_hat,
                                            &L2_hat,
                                            &L3_hat);
          const real_t v = visc_.eval(zisa::sqrt(absk2));
          dudt_hat(0, i, j, k)
              = absk2 == 0 ? 0 : -L1_hat + v * u_hat(0, i, j, k);
          dudt_hat(1, i, j, k)
              = absk2 == 0 ? 0 : -L2_hat + v * u_hat(1, i, j, k);
          dudt_hat(2, i, j, k)
              = absk2 == 0 ? 0 : -L3_hat + v * u_hat(2, i, j, k);
          if (has_tracer_) {
            complex_t L4_hat;
            advection_3d(k3,
                         k2,
                         k1,
                         stride_B,
                         idx_B,
                         B_hat_.raw() + 6 * stride_B,
                         &L4_hat);
            dudt_hat(3, i, j, k) = -L4_hat + v * u_hat(3, i, j, k);
          }
        }
      }
    }
  }

#if ZISA_HAS_CUDA
  void computeDudt_cuda(const zisa::array_view<complex_t, 4> &dudt_hat,
                        const zisa::array_const_view<complex_t, 4> &u_hat) {
    const zisa::int_t i_base = grid_.i_fourier(0, comm_);
    const zisa::int_t j_base = grid_.j_fourier(0, comm_);
    const zisa::int_t k_base = grid_.k_fourier(0, comm_);
    const auto shape_phys = grid_.shape_phys(1);
    if (has_tracer_) {
      incompressible_euler_mpi_3d_tracer_cuda(B_hat_,
                                              u_hat,
                                              dudt_hat,
                                              visc_,
                                              forcing_,
                                              i_base,
                                              j_base,
                                              k_base,
                                              shape_phys);
    } else {
      incompressible_euler_mpi_3d_cuda(B_hat_,
                                       u_hat,
                                       dudt_hat,
                                       visc_,
                                       forcing_,
                                       i_base,
                                       j_base,
                                       k_base,
                                       shape_phys);
    }
  }
#endif
};

}

#endif
