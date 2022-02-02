#ifndef AZEBAN_EQUATIONS_INCOMPRESSIBLE_EULER_MPI_HPP_
#define AZEBAN_EQUATIONS_INCOMPRESSIBLE_EULER_MPI_HPP_

#include <azeban/equations/equation.hpp>
#include <azeban/forcing/no_forcing.hpp>
#include <azeban/memory/workspace.hpp>
#include <azeban/operations/fft.hpp>
#include <mpi.h>

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

  virtual void *get_fft_work_area() override;

protected:
  using super::grid_;
  MPI_Comm comm_;
  zisa::device_type device_;
  bool has_tracer_;
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
  zisa::array_view<complex_t, 1> u_yz_pre_;
  zisa::array_view<complex_t, 1> u_yz_comm_;
  zisa::array_view<complex_t, dim_v + 1> u_yz_trans_;
  zisa::array_view<complex_t, dim_v + 1> u_yz_trans_pad_;
  zisa::array_view<real_t, dim_v + 1> u_xyz_trans_;
  zisa::array_view<real_t, dim_v + 1> B_xyz_trans_;

  static zisa::array_view<complex_t, dim_v>
  component(const zisa::array_view<complex_t, dim_v + 1> &arr, int dim);
  static zisa::array_const_view<complex_t, dim_v>
  component(const zisa::array_const_view<complex_t, dim_v + 1> &arr, int dim);
  static zisa::array_view<complex_t, dim_v>
  component(zisa::array<complex_t, dim_v + 1> &arr, int dim);
  static zisa::array_const_view<complex_t, dim_v>
  component(const zisa::array<complex_t, dim_v + 1> &arr, int dim);

  void compute_B_hat(const zisa::array_const_view<complex_t, dim_v + 1> &u_hat);

private:
  void
  compute_u_hat_pad(const zisa::array_const_view<complex_t, dim_v + 1> &u_hat);
  void compute_u_yz();
  void compute_u_yz_pre();
  void compute_u_yz_pre_cpu();
  void compute_u_yz_comm();
  void compute_u_yz_trans();
  void compute_u_yz_trans_cpu();
  void compute_u_yz_trans_pad();
  void compute_u_xyz_trans();
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

  using super::n_vars;

protected:
  using super::compute_B_hat;
  using super::device_;
  using super::grid_;

private:
  SpectralViscosity visc_;
  Forcing forcing_;
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

  using super::n_vars;

protected:
  using super::compute_B_hat;
  using super::device_;
  using super::grid_;

private:
  SpectralViscosity visc_;
  Forcing forcing_;
};

}

#endif
