#include <azeban/cuda/equations/incompressible_euler_mpi_cuda.hpp>
#include <azeban/cuda/equations/incompressible_euler_mpi_cuda_impl.cuh>
#include <azeban/forcing/no_forcing.hpp>
#include <azeban/forcing/sinusoidal.hpp>

namespace azeban {

#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(VISC, FORCING)        \
  template void incompressible_euler_mpi_2d_cuda<VISC, FORCING>(               \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      unsigned long,                                                           \
      unsigned long,                                                           \
      const zisa::shape_t<3> &);                                               \
  template void incompressible_euler_mpi_3d_cuda<VISC, FORCING>(               \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      unsigned long,                                                           \
      unsigned long,                                                           \
      unsigned long,                                                           \
      const zisa::shape_t<4> &);                                               \
  template void incompressible_euler_mpi_2d_tracer_cuda<VISC, FORCING>(        \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      unsigned long,                                                           \
      unsigned long,                                                           \
      const zisa::shape_t<3> &);                                               \
  template void incompressible_euler_mpi_3d_tracer_cuda<VISC, FORCING>(        \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      unsigned long,                                                           \
      unsigned long,                                                           \
      unsigned long,                                                           \
      const zisa::shape_t<4> &);

AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(Step1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    Step1D, WhiteNoise<curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    Step1D, WhiteNoise<curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(Step1D,
                                                 WhiteNoise<std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(Step1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(SmoothCutoff1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    SmoothCutoff1D, WhiteNoise<curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    SmoothCutoff1D, WhiteNoise<curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(SmoothCutoff1D,
                                                 WhiteNoise<std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(SmoothCutoff1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(Quadratic, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    Quadratic, WhiteNoise<curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    Quadratic, WhiteNoise<curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(Quadratic,
                                                 WhiteNoise<std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(Quadratic, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(NoViscosity, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    NoViscosity, WhiteNoise<curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(
    NoViscosity, WhiteNoise<curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(NoViscosity,
                                                 WhiteNoise<std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA(NoViscosity, Sinusoidal)

#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_MPI_CUDA

}
