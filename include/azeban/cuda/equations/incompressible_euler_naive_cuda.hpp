#ifndef INCOMPRESSIBLE_EULER_NAIVE_CUDA_H_
#define INCOMPRESSIBLE_EULER_NAIVE_CUDA_H_

#include <azeban/config.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/grid.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
void incompressible_euler_naive_compute_B_cuda(
    const zisa::array_view<real_t, Dim + 1> &B,
    const zisa::array_const_view<real_t, Dim + 1> &u,
    const Grid<Dim> &grid);

template <typename SpectralViscosity>
void incompressible_euler_naive_2d_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_view<complex_t, 3> &u_hat,
    const SpectralViscosity &visc);

template <typename SpectralViscosity>
void incompressible_euler_naive_3d_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_view<complex_t, 4> &u_hat,
    const SpectralViscosity &visc);

#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_NAIVE_CUDA(TYPE)               \
  extern template void incompressible_euler_naive_2d_cuda<TYPE>(               \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const TYPE &);                                                           \
  extern template void incompressible_euler_naive_3d_cuda<TYPE>(               \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const TYPE &);

AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_NAIVE_CUDA(Step1D)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_NAIVE_CUDA(SmoothCutoff1D)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_NAIVE_CUDA(Quadratic)

#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_NAIVE_CUDA

}

#endif
