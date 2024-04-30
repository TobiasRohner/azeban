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
#include <azeban/forcing/sinusoidal.hpp>
#include <azeban/forcing/white_noise.hpp>
#include <azeban/forcing/boussinesq.hpp>
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
    Forcing &forcing,
    real_t t,
    real_t dt);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_3d_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    real_t t,
    real_t dt);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_2d_tracer_cuda(
    const zisa::array_const_view<complex_t, 3> &B_hat,
    const zisa::array_const_view<complex_t, 3> &u_hat,
    const zisa::array_view<complex_t, 3> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    real_t t,
    real_t dt);

template <typename SpectralViscosity, typename Forcing>
void incompressible_euler_3d_tracer_cuda(
    const zisa::array_const_view<complex_t, 4> &B_hat,
    const zisa::array_const_view<complex_t, 4> &u_hat,
    const zisa::array_view<complex_t, 4> &dudt_hat,
    const SpectralViscosity &visc,
    Forcing &forcing,
    real_t t,
    real_t dt);

#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(VISC, FORCING)         \
  extern template void incompressible_euler_2d_cuda<VISC, FORCING>(            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      real_t,                                                                  \
      real_t);                                                                 \
  extern template void incompressible_euler_2d_tracer_cuda<VISC, FORCING>(     \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      real_t,                                                                  \
      real_t);
#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(VISC, FORCING)         \
  extern template void incompressible_euler_3d_cuda<VISC, FORCING>(            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      real_t,                                                                  \
      real_t);                                                                 \
  extern template void incompressible_euler_3d_tracer_cuda<VISC, FORCING>(     \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const VISC &,                                                            \
      FORCING &,                                                               \
      real_t,                                                                  \
      real_t);

#define COMMA ,

AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(Step1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(
    Step1D, WhiteNoise<2 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(
    Step1D, WhiteNoise<2 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(
    Step1D, WhiteNoise<2 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(Step1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(Step1D, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(SmoothCutoff1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(
    SmoothCutoff1D, WhiteNoise<2 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(
    SmoothCutoff1D, WhiteNoise<2 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(
    SmoothCutoff1D, WhiteNoise<2 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(SmoothCutoff1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(SmoothCutoff1D, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(Quadratic, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(
    Quadratic, WhiteNoise<2 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(
    Quadratic, WhiteNoise<2 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(
    Quadratic, WhiteNoise<2 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(Quadratic, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(Quadratic, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(NoViscosity, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(
    NoViscosity, WhiteNoise<2 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(
    NoViscosity, WhiteNoise<2 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(
    NoViscosity, WhiteNoise<2 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(NoViscosity, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D(NoViscosity, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(Step1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(
    Step1D, WhiteNoise<3 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(
    Step1D, WhiteNoise<3 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(
    Step1D, WhiteNoise<3 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(Step1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(Step1D, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(SmoothCutoff1D, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(
    SmoothCutoff1D, WhiteNoise<3 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(
    SmoothCutoff1D, WhiteNoise<3 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(
    SmoothCutoff1D, WhiteNoise<3 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(SmoothCutoff1D, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(SmoothCutoff1D, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(Quadratic, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(
    Quadratic, WhiteNoise<3 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(
    Quadratic, WhiteNoise<3 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(
    Quadratic, WhiteNoise<3 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(Quadratic, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(Quadratic, Boussinesq)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(NoViscosity, NoForcing)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(
    NoViscosity, WhiteNoise<3 COMMA curandStateMRG32k3a_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(
    NoViscosity, WhiteNoise<3 COMMA curandStateXORWOW_t>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(
    NoViscosity, WhiteNoise<3 COMMA std::mt19937>)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(NoViscosity, Sinusoidal)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D(NoViscosity, Boussinesq)

#undef COMMA

#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_2D
#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA_3D

}

#endif
