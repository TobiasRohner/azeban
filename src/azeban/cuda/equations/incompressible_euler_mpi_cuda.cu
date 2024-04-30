#include <azeban/cuda/equations/incompressible_euler_mpi_cuda.hpp>
#include <azeban/cuda/equations/incompressible_euler_mpi_cuda_impl.cuh>
#include <azeban/forcing/no_forcing.hpp>
#include <azeban/forcing/sinusoidal.hpp>

namespace azeban {

#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(VISC, FORCING)     \
  template void incompressible_euler_mpi_2d_cuda<VISC, FORCING>(               \
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
  template void incompressible_euler_mpi_2d_tracer_cuda<VISC, FORCING>(        \
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
  template void incompressible_euler_mpi_3d_cuda<VISC, FORCING>(               \
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
  template void incompressible_euler_mpi_3d_tracer_cuda<VISC, FORCING>(        \
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
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(Step1D, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(SmoothCutoff1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    SmoothCutoff1D, WhiteNoise<2 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    SmoothCutoff1D, WhiteNoise<2 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    SmoothCutoff1D, WhiteNoise<2 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(SmoothCutoff1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(SmoothCutoff1D, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(Quadratic, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    Quadratic, WhiteNoise<2 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    Quadratic, WhiteNoise<2 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    Quadratic, WhiteNoise<2 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(Quadratic, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(Quadratic, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(NoViscosity, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    NoViscosity, WhiteNoise<2 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    NoViscosity, WhiteNoise<2 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(
    NoViscosity, WhiteNoise<2 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(NoViscosity, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D(NoViscosity, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(Step1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    Step1D, WhiteNoise<3 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    Step1D, WhiteNoise<3 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    Step1D, WhiteNoise<3 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(Step1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(Step1D, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(SmoothCutoff1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    SmoothCutoff1D, WhiteNoise<3 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    SmoothCutoff1D, WhiteNoise<3 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    SmoothCutoff1D, WhiteNoise<3 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(SmoothCutoff1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(SmoothCutoff1D, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(Quadratic, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    Quadratic, WhiteNoise<3 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    Quadratic, WhiteNoise<3 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    Quadratic, WhiteNoise<3 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(Quadratic, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(Quadratic, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(NoViscosity, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    NoViscosity, WhiteNoise<3 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    NoViscosity, WhiteNoise<3 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(
    NoViscosity, WhiteNoise<3 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(NoViscosity, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D(NoViscosity, Boussinesq)

#undef COMMA

#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_2D
#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA_3D

}
