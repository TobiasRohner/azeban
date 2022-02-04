#ifndef AZEBAN_CUDA_EQUATIONS_INCOMPRESSIBLE_EULER_MPI_CUDA_HPP_
#define AZEBAN_CUDA_EQUATIONS_INCOMPRESSIBLE_EULER_MPI_CUDA_HPP_

#include <azeban/config.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/forcing/no_forcing.hpp>
#include <azeban/forcing/white_noise.hpp>
#include <azeban/grid.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_mpi_2d_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    unsigned i_base,
    unsigned j_base,
    const zisa::shape_t<3> &shape_phys);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_mpi_3d_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    unsigned i_base,
    unsigned j_base,
    unsigned k_base,
    const zisa::shape_t<4> &shape_phys);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_mpi_2d_tracer_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    unsigned i_base,
    unsigned j_base,
    const zisa::shape_t<3> &shape_phys);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_mpi_3d_tracer_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    unsigned i_base,
    unsigned j_base,
    unsigned k_base,
    const zisa::shape_t<4> &shape_phys);

#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(VISC, FORCING)        \
  extern template void incompressible_euler_mpi_2d_cuda<VISC, FORCING>(        \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      unsigned,                                                                \
      unsigned,                                                                \
      const zisa::shape_t<3> &);                                               \
  extern template void incompressible_euler_mpi_3d_cuda<VISC, FORCING>(        \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      unsigned,                                                                \
      unsigned,                                                                \
      unsigned,                                                                \
      const zisa::shape_t<4> &);                                               \
  extern template void incompressible_euler_mpi_2d_tracer_cuda<VISC, FORCING>( \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      unsigned,                                                                \
      unsigned,                                                                \
      const zisa::shape_t<3> &);                                               \
  extern template void incompressible_euler_mpi_3d_tracer_cuda<VISC, FORCING>( \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      unsigned,                                                                \
      unsigned,                                                                \
      unsigned,                                                                \
      const zisa::shape_t<4> &);

AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(Step1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    Step1D, WhiteNoise<curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    Step1D, WhiteNoise<curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(Step1D,
                                                 WhiteNoise<std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(SmoothCutoff1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    SmoothCutoff1D, WhiteNoise<curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    SmoothCutoff1D, WhiteNoise<curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(SmoothCutoff1D,
                                                 WhiteNoise<std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(Quadratic, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    Quadratic, WhiteNoise<curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    Quadratic, WhiteNoise<curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(Quadratic,
                                                 WhiteNoise<std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(NoViscosity, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    NoViscosity, WhiteNoise<curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    NoViscosity, WhiteNoise<curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(NoViscosity,
                                                 WhiteNoise<std::mt19937>)

#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA

}

#endif
