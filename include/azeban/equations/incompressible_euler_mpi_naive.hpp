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
#ifndef AZEBAN_EQUATIONS_INCOMPRESSIBLE_EULER_MPI_NAIVE_HPP_
#define AZEBAN_EQUATIONS_INCOMPRESSIBLE_EULER_MPI_NAIVE_HPP_

#include "advection_functions.hpp"
#include "equation.hpp"
#include "incompressible_euler_functions.hpp"
#include <azeban/config.hpp>
#include <azeban/cuda/equations/incompressible_euler_cuda.hpp>
#include <azeban/forcing/no_forcing.hpp>
#include <azeban/mpi/communicator.hpp>
#include <azeban/operations/fft_mpi_factory.hpp>
#include <azeban/operations/norm.hpp>
#include <azeban/profiler.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace azeban {

template <int Dim>
class IncompressibleEuler_MPI_Naive_Base : public Equation<Dim> {
  using super = Equation<Dim>;
  static_assert(Dim == 2 || Dim == 3,
                "Incompressible Euler is only implemented for 2D and 3D");

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = Dim;

  IncompressibleEuler_MPI_Naive_Base(const Grid<dim_v> &grid,
                                     const Communicator *comm,
                                     bool has_tracer = false);
  IncompressibleEuler_MPI_Naive_Base(const IncompressibleEuler_MPI_Naive_Base &)
      = delete;
  IncompressibleEuler_MPI_Naive_Base(IncompressibleEuler_MPI_Naive_Base &&)
      = default;
  virtual ~IncompressibleEuler_MPI_Naive_Base() = default;
  IncompressibleEuler_MPI_Naive_Base &
  operator=(const IncompressibleEuler_MPI_Naive_Base &)
      = delete;
  IncompressibleEuler_MPI_Naive_Base &
  operator=(IncompressibleEuler_MPI_Naive_Base &&)
      = default;

  virtual int n_vars() const override { return dim_v + (has_tracer_ ? 1 : 0); }

  virtual void *get_fft_work_area() const override {
    return fft_B_->get_work_area();
  }

protected:
  using super::grid_;
  const Communicator *comm_;
  zisa::array<complex_t, dim_v + 1> u_hat_partial_pad_;
  zisa::array<complex_t, dim_v + 1> h_u_hat_pad_;
  zisa::array<complex_t, dim_v + 1> d_u_hat_pad_;
  zisa::array<real_t, dim_v + 1> u_pad_;
  zisa::array<real_t, dim_v + 1> B_pad_;
  zisa::array<complex_t, dim_v + 1> d_B_hat_pad_;
  zisa::array<complex_t, dim_v + 1> h_B_hat_pad_;
  zisa::array<complex_t, dim_v + 1> B_hat_partial_pad_;
  zisa::array<complex_t, dim_v + 1> B_hat_;
  std::shared_ptr<FFT<dim_v>> fft_u_;
  std::shared_ptr<FFT<dim_v>> fft_B_;
  bool has_tracer_;

  static zisa::array_view<complex_t, dim_v>
  component(const zisa::array_view<complex_t, dim_v + 1> &arr, int dim);
  static zisa::array_const_view<complex_t, dim_v>
  component(const zisa::array_const_view<complex_t, dim_v + 1> &arr, int dim);
  static zisa::array_view<complex_t, dim_v>
  component(zisa::array<complex_t, dim_v + 1> &arr, int dim);
  static zisa::array_const_view<complex_t, dim_v>
  component(const zisa::array<complex_t, dim_v + 1> &arr, int dim);

  void computeB();
  void pad_u_hat(const zisa::array_const_view<complex_t, dim_v + 1> &u_hat);
  void unpad_B_hat();
};

template <int Dim, typename SpectralViscosity, typename Forcing = NoForcing>
class IncompressibleEuler_MPI_Naive {
  static_assert(Dim == 2 || Dim == 3,
                "Incompressible Euler is only implemented for 2D and 3D");
};

template <typename SpectralViscosity, typename Forcing>
class IncompressibleEuler_MPI_Naive<2, SpectralViscosity, Forcing>
    : public IncompressibleEuler_MPI_Naive_Base<2> {
  using super = IncompressibleEuler_MPI_Naive_Base<2>;

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = 2;

  template <bool enable = std::is_same_v<Forcing, NoForcing>,
            typename = std::enable_if_t<enable>>
  IncompressibleEuler_MPI_Naive(const Grid<2> &grid,
                                const Communicator *comm,
                                const SpectralViscosity &visc,
                                bool has_tracer = false)
      : IncompressibleEuler_MPI_Naive(
          grid, comm, visc, NoForcing{}, has_tracer) {}
  IncompressibleEuler_MPI_Naive(const Grid<2> &grid,
                                const Communicator *comm,
                                const SpectralViscosity &visc,
                                const Forcing &forcing,
                                bool has_tracer = false)
      : super(grid, comm, has_tracer),
        u_max_(0),
        visc_(visc),
        forcing_(forcing) {}
  IncompressibleEuler_MPI_Naive(const IncompressibleEuler_MPI_Naive &) = delete;
  IncompressibleEuler_MPI_Naive(IncompressibleEuler_MPI_Naive &&) = default;
  virtual ~IncompressibleEuler_MPI_Naive() = default;
  IncompressibleEuler_MPI_Naive &
  operator=(const IncompressibleEuler_MPI_Naive &)
      = delete;
  IncompressibleEuler_MPI_Naive &operator=(IncompressibleEuler_MPI_Naive &&)
      = default;

  virtual void dudt(const zisa::array_view<complex_t, 3> &dudt_hat,
                    const zisa::array_const_view<complex_t, 3> &u_hat,
                    real_t t,
                    real_t dt) override {
    LOG_ERR_IF(u_hat.memory_location() != zisa::device_type::cpu,
               "Euler MPI needs CPU arrays");
    LOG_ERR_IF(u_hat.shape(0) != h_u_hat_pad_.shape(0),
               "Wrong number of variables");
    LOG_ERR_IF(dudt_hat.memory_location() != zisa::device_type::cpu,
               "Euler MPI needs CPU arrays");
    LOG_ERR_IF(dudt_hat.shape(0) != h_u_hat_pad_.shape(0),
               "Wrong number of variables");
    ProfileHost profile("IncompressibleEuler_MPI_Naive::dudt");
    pad_u_hat(u_hat);
    zisa::copy(d_u_hat_pad_, h_u_hat_pad_);
    fft_u_->backward();
    real_t max_u_loc = max_norm(fft_u_->u());
    MPI_Allreduce(&max_u_loc,
                  &u_max_,
                  1,
                  mpi_type<real_t>(),
                  MPI_MAX,
                  comm_->get_mpi_comm());
    computeB();
    fft_B_->forward();
    zisa::copy(h_B_hat_pad_, d_B_hat_pad_);
    unpad_B_hat();
    computeDudt(dudt_hat, u_hat, t, dt);
  }

  virtual real_t dt() const override {
    return zisa::pow<1>(grid_.N_phys) / u_max_;
  }

  using super::n_vars;

protected:
  using super::B_hat_;
  using super::B_pad_;
  using super::comm_;
  using super::d_B_hat_pad_;
  using super::d_u_hat_pad_;
  using super::fft_B_;
  using super::fft_u_;
  using super::grid_;
  using super::h_B_hat_pad_;
  using super::h_u_hat_pad_;
  using super::has_tracer_;
  using super::u_pad_;

  using super::component;
  using super::computeB;
  using super::pad_u_hat;
  using super::unpad_B_hat;

private:
  real_t u_max_;
  SpectralViscosity visc_;
  Forcing forcing_;

  void computeDudt(const zisa::array_view<complex_t, 3> &dudt_hat,
                   const zisa::array_const_view<complex_t, 3> &u_hat,
                   real_t t,
                   real_t dt) {
    ProfileHost profile("IncompressibleEuler_MPI_Naive::computeDudt");
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
	const complex_t u = u_hat(0, i, j);
	const complex_t v = u_hat(1, i, j);
	const complex_t rho = has_tracer_ ? u_hat(2, i, j) : 1;
        complex_t force1, force2;
        forcing_(t, dt, u, v, rho, j_, i_, &force1, &force2);
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
        const real_t nu = visc_.eval(zisa::sqrt(absk2));
        dudt_hat(0, i, j) = absk2 == 0 ? 0 : -L1_hat + nu * u;
        dudt_hat(1, i, j) = absk2 == 0 ? 0 : -L2_hat + nu * v;
        if (has_tracer_) {
          complex_t L3_hat;
          advection_2d(
              k2, k1, stride_B, idx_B, B_hat_.raw() + 3 * stride_B, &L3_hat);
          dudt_hat(2, i, j) = -L3_hat + nu * rho;
        }
      }
    }
  }
};

template <typename SpectralViscosity, typename Forcing>
class IncompressibleEuler_MPI_Naive<3, SpectralViscosity, Forcing>
    : public IncompressibleEuler_MPI_Naive_Base<3> {
  using super = IncompressibleEuler_MPI_Naive_Base<3>;

public:
  using scalar_t = complex_t;
  static constexpr int dim_v = 3;

  template <bool enable = std::is_same_v<Forcing, NoForcing>,
            typename = std::enable_if_t<enable>>
  IncompressibleEuler_MPI_Naive(const Grid<3> &grid,
                                const Communicator *comm,
                                const SpectralViscosity &visc,
                                bool has_tracer = false)
      : IncompressibleEuler_MPI_Naive(
          grid, comm, visc, NoForcing{}, has_tracer) {}
  IncompressibleEuler_MPI_Naive(const Grid<3> &grid,
                                const Communicator *comm,
                                const SpectralViscosity &visc,
                                const Forcing &forcing,
                                bool has_tracer = false)
      : super(grid, comm, has_tracer),
        u_max_(0),
        visc_(visc),
        forcing_(forcing) {}
  IncompressibleEuler_MPI_Naive(const IncompressibleEuler_MPI_Naive &) = delete;
  IncompressibleEuler_MPI_Naive(IncompressibleEuler_MPI_Naive &&) = default;
  virtual ~IncompressibleEuler_MPI_Naive() = default;
  IncompressibleEuler_MPI_Naive &
  operator=(const IncompressibleEuler_MPI_Naive &)
      = delete;
  IncompressibleEuler_MPI_Naive &operator=(IncompressibleEuler_MPI_Naive &&)
      = default;

  virtual void dudt(const zisa::array_view<complex_t, 4> &dudt_hat,
                    const zisa::array_const_view<complex_t, 4> &u_hat,
                    real_t t,
                    real_t dt) override {
    LOG_ERR_IF(u_hat.memory_location() != zisa::device_type::cpu,
               "Euler MPI needs CPU arrays");
    LOG_ERR_IF(u_hat.shape(0) != h_u_hat_pad_.shape(0),
               "Wrong number of variables");
    LOG_ERR_IF(dudt_hat.memory_location() != zisa::device_type::cpu,
               "Euler MPI needs CPU arrays");
    LOG_ERR_IF(dudt_hat.shape(0) != h_u_hat_pad_.shape(0),
               "Wrong number of variables");
    ProfileHost profile("IncompressibleEuler_MPI_Naive::dudt");
    pad_u_hat(u_hat);
    zisa::copy(d_u_hat_pad_, h_u_hat_pad_);
    fft_u_->backward();
    real_t max_u_loc = max_norm(fft_u_->u());
    MPI_Allreduce(&max_u_loc,
                  &u_max_,
                  1,
                  mpi_type<real_t>(),
                  MPI_SUM,
                  comm_->get_mpi_comm());
    computeB();
    fft_B_->forward();
    zisa::copy(h_B_hat_pad_, d_B_hat_pad_);
    unpad_B_hat();
    computeDudt(dudt_hat, u_hat, t, dt);
  }

  virtual real_t dt() const override {
    return zisa::pow<2>(grid_.N_phys) / u_max_;
  }

  using super::n_vars;

protected:
  using super::B_hat_;
  using super::B_pad_;
  using super::comm_;
  using super::d_B_hat_pad_;
  using super::d_u_hat_pad_;
  using super::fft_B_;
  using super::fft_u_;
  using super::grid_;
  using super::h_B_hat_pad_;
  using super::h_u_hat_pad_;
  using super::has_tracer_;
  using super::u_pad_;

  using super::component;
  using super::computeB;
  using super::pad_u_hat;
  using super::unpad_B_hat;

private:
  real_t u_max_;
  SpectralViscosity visc_;
  Forcing forcing_;

  void computeDudt(const zisa::array_view<complex_t, 4> &dudt_hat,
                   const zisa::array_const_view<complex_t, 4> &u_hat,
                   real_t t,
                   real_t dt) {
    ProfileHost profile("IncompressibleEuler_MPI_Naive::computeDudt");
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
	  const complex_t u = u_hat(0, i, j, k);
	  const complex_t v = u_hat(1, i, j, k);
	  const complex_t w = u_hat(2, i, j, k);
	  const complex_t rho = has_tracer_ ? u_hat(3, i, j, k) : 1;
          complex_t force1, force2, force3;
          forcing_(t, dt, u, v, w, rho, k_, j_, i_, &force1, &force2, &force3);
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
          const real_t nu = visc_.eval(zisa::sqrt(absk2));
          dudt_hat(0, i, j, k)
              = absk2 == 0 ? 0 : -L1_hat + nu * u;
          dudt_hat(1, i, j, k)
              = absk2 == 0 ? 0 : -L2_hat + nu * v;
          dudt_hat(2, i, j, k)
              = absk2 == 0 ? 0 : -L3_hat + nu * w;
          if (has_tracer_) {
            complex_t L4_hat;
            advection_3d(k3,
                         k2,
                         k1,
                         stride_B,
                         idx_B,
                         B_hat_.raw() + 6 * stride_B,
                         &L4_hat);
            dudt_hat(3, i, j, k) = -L4_hat + nu * rho;
          }
        }
      }
    }
  }
};

}

#endif
