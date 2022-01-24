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
#include <azeban/cuda/equations/incompressible_euler_cuda.hpp>
#include <azeban/cuda/equations/incompressible_euler_cuda_impl.cuh>
#include <azeban/forcing/no_forcing.hpp>

namespace azeban {

#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_COMPUTE_B_CUDA(DIM)            \
  template void incompressible_euler_compute_B_cuda<DIM>(                      \
      const zisa::array_view<real_t, DIM + 1> &,                               \
      const zisa::array_const_view<real_t, DIM + 1> &,                         \
      const Grid<DIM> &);                                                      \
  template void incompressible_euler_compute_B_tracer_cuda<DIM>(               \
      const zisa::array_view<real_t, DIM + 1> &,                               \
      const zisa::array_const_view<real_t, DIM + 1> &,                         \
      const Grid<DIM> &);

AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_COMPUTE_B_CUDA(2)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_COMPUTE_B_CUDA(3)

#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_COMPUTE_B_CUDA

#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(VISC, FORCING)            \
  template void incompressible_euler_2d_cuda<VISC, FORCING>(                   \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const VISC &,                                                            \
      FORCING &);                                                              \
  template void incompressible_euler_3d_cuda<VISC, FORCING>(                   \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const VISC &,                                                            \
      FORCING &);                                                              \
  template void incompressible_euler_2d_tracer_cuda<VISC, FORCING>(            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const VISC &,                                                            \
      FORCING &);                                                              \
  template void incompressible_euler_3d_tracer_cuda<VISC, FORCING>(            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const VISC &,                                                            \
      FORCING &);

AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(Step1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(SmoothCutoff1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(Quadratic, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(NoViscosity, NoForcing)

#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA

}
