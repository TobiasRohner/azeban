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
#ifndef BURGERS_CUDA_H_
#define BURGERS_CUDS_H_

#include <azeban/config.hpp>
#include <azeban/equations/spectral_viscosity.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename SpectralViscosity>
void burgers_cuda(const zisa::array_view<complex_t, 2> &dudt,
                  const zisa::array_const_view<complex_t, 2> &u,
                  const zisa::array_const_view<complex_t, 2> &u_squared,
                  const SpectralViscosity &visc);

#define AZEBAN_INSTANTIATE_BURGERS_CUDA(TYPE)                                  \
  extern template void burgers_cuda<TYPE>(                                     \
      const zisa::array_view<complex_t, 2> &,                                  \
      const zisa::array_const_view<complex_t, 2> &,                            \
      const zisa::array_const_view<complex_t, 2> &,                            \
      const TYPE &);

AZEBAN_INSTANTIATE_BURGERS_CUDA(Step1D)
AZEBAN_INSTANTIATE_BURGERS_CUDA(SmoothCutoff1D)
AZEBAN_INSTANTIATE_BURGERS_CUDA(Quadratic)
AZEBAN_INSTANTIATE_BURGERS_CUDA(NoViscosity)

#undef AZEBAN_INSTANTIATE_BURGERS_CUDA

}

#endif
