#ifndef AZEBAN_CUDA_EQUATIONS_INCOMPRESSIBLE_EULER_MPI_CUDA_HPP_
#define AZEBAN_CUDA_EQUATIONS_INCOMPRESSIBLE_EULER_MPI_CUDA_HPP_

#include <azeban/config.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/forcing/no_forcing.hpp>
#include <azeban/forcing/sinusoidal.hpp>
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
    real_t t,
    real_t dt,
    unsigned long i_base,
    unsigned long j_base,
    const zisa::shape_t<3> &shape_phys);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_mpi_3d_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    real_t t,
    real_t dt,
    unsigned long i_base,
    unsigned long j_base,
    unsigned long k_base,
    const zisa::shape_t<4> &shape_phys);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_mpi_2d_tracer_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    real_t t,
    real_t dt,
    unsigned long i_base,
    unsigned long j_base,
    const zisa::shape_t<3> &shape_phys);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_mpi_3d_tracer_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    real_t t,
    real_t dt,
    unsigned long i_base,
    unsigned long j_bae,
    unsigned long k_base,
    const zisa::shape_t<4> &shape_phys);

#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(VISC, FORCING)     \
  extern template void incompressible_euler_mpi_2d_cuda<VISC, FORCING>(        \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      real_t,                                                                  \
      real_t,                                                                  \
      unsigned long,                                                           \
      unsigned long,                                                           \
      const zisa::shape_t<3> &);                                               \
  extern template void incompressible_euler_mpi_2d_tracer_cuda<VISC, FORCING>( \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      real_t,                                                                  \
      real_t,                                                                  \
      unsigned long,                                                           \
      unsigned long,                                                           \
      const zisa::shape_t<3> &);
#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(VISC, FORCING)     \
  extern template void incompressible_euler_mpi_3d_cuda<VISC, FORCING>(        \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      real_t,                                                                  \
      real_t,                                                                  \
      unsigned long,                                                           \
      unsigned long,                                                           \
      unsigned long,                                                           \
      const zisa::shape_t<4> &);                                               \
  extern template void incompressible_euler_mpi_3d_tracer_cuda<VISC, FORCING>( \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      real_t,                                                                  \
      real_t,                                                                  \
      unsigned long,                                                           \
      unsigned long,                                                           \
      unsigned long,                                                           \
      const zisa::shape_t<4> &);

#define COMMA ,

AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(Step1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    Step1D, WhiteNoise<2 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    Step1D, WhiteNoise<2 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    Step1D, WhiteNoise<2 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(Step1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(SmoothCutoff1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    SmoothCutoff1D, WhiteNoise<2 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    SmoothCutoff1D, WhiteNoise<2 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    SmoothCutoff1D, WhiteNoise<2 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(SmoothCutoff1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(Quadratic, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    Quadratic, WhiteNoise<2 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    Quadratic, WhiteNoise<2 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    Quadratic, WhiteNoise<2 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(Quadratic, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(NoViscosity, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    NoViscosity, WhiteNoise<2 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    NoViscosity, WhiteNoise<2 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    NoViscosity, WhiteNoise<2 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(NoViscosity, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(Step1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    Step1D, WhiteNoise<3 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    Step1D, WhiteNoise<3 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    Step1D, WhiteNoise<3 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(Step1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(SmoothCutoff1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    SmoothCutoff1D, WhiteNoise<3 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    SmoothCutoff1D, WhiteNoise<3 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    SmoothCutoff1D, WhiteNoise<3 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(SmoothCutoff1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(Quadratic, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    Quadratic, WhiteNoise<3 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    Quadratic, WhiteNoise<3 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    Quadratic, WhiteNoise<3 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(Quadratic, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(NoViscosity, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    NoViscosity, WhiteNoise<3 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    NoViscosity, WhiteNoise<3 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    NoViscosity, WhiteNoise<3 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(NoViscosity, Sinusoidal)

#undef COMMA

#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D
#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D

}

#endif
