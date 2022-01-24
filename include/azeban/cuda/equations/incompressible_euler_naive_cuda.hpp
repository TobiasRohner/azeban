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
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_NAIVE_CUDA(NoViscosity)

#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_NAIVE_CUDA

}

#endif
