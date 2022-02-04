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
#ifndef AZEBAN_CUDA_EQUATIONS_INCOMPRESSIBLE_EULER_CUDA_HPP_
#define AZEBAN_CUDA_EQUATIONS_INCOMPRESSIBLE_EULER_CUDA_HPP_

#include <azeban/config.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <azeban/forcing/no_forcing.hpp>
#include <azeban/forcing/white_noise.hpp>
#include <azeban/grid.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
void incompressible_euler_compute_B_cuda(
    const zisa::array_view<real_t, Dim + 1> &B,
    const zisa::array_const_view<real_t, Dim + 1> &u,
    const Grid<Dim> &grid);

template <int Dim>
void incompressible_euler_compute_B_tracer_cuda(
    const zisa::array_view<real_t, Dim + 1> &B,
    const zisa::array_const_view<real_t, Dim + 1> &u,
    const Grid<Dim> &grid);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_2d_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_3d_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_2d_tracer_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_3d_tracer_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing);

#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(VISC, FORCING)            \
  extern template void incompressible_euler_2d_cuda<VISC, FORCING>(            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const VISC &,                                                            \
      FORCING &);                                                              \
  extern template void incompressible_euler_3d_cuda<VISC, FORCING>(            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const VISC &,                                                            \
      FORCING &);                                                              \
  extern template void incompressible_euler_2d_tracer_cuda<VISC, FORCING>(     \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const VISC &,                                                            \
      FORCING &);                                                              \
  extern template void incompressible_euler_3d_tracer_cuda<VISC, FORCING>(     \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const VISC &,                                                            \
      FORCING &);

AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(Step1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(Step1D,
                                             WhiteNoise<curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(Step1D,
                                             WhiteNoise<curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(Step1D, WhiteNoise<std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(SmoothCutoff1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(SmoothCutoff1D,
                                             WhiteNoise<curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(SmoothCutoff1D,
                                             WhiteNoise<curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(SmoothCutoff1D,
                                             WhiteNoise<std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(Quadratic, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(Quadratic,
                                             WhiteNoise<curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(Quadratic,
                                             WhiteNoise<curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(Quadratic,
                                             WhiteNoise<std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(NoViscosity, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(NoViscosity,
                                             WhiteNoise<curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(NoViscosity,
                                             WhiteNoise<curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(NoViscosity,
                                             WhiteNoise<std::mt19937>)

#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA

}

#endif
