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
#ifndef CONVOLVE_H_
#define CONVOLVE_H_

#include <azeban/operations/copy_to_padded.hpp>
#include <azeban/operations/fft.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/convolve_cuda.hpp>
#endif

namespace azeban {

namespace detail {

template <int Dim>
void scale_and_square(const zisa::array_view<real_t, Dim> &u, real_t scale) {
  const zisa::shape_t<1> flat_shape{zisa::product(u.shape())};
  const zisa::array_view<real_t, 1> flat(
      flat_shape, u.raw(), u.memory_location());
  if (flat.memory_location() == zisa::device_type::cpu) {
    for (zisa::int_t i = 0; i < flat.shape(0); ++i) {
      const real_t ui_scaled = scale * u[i];
      flat[i] = ui_scaled * ui_scaled;
    }
  }
#ifdef ZISA_HAS_CUDA
  else if (flat.memory_location() == zisa::device_type::cuda) {
    scale_and_square_cuda(flat, scale);
  }
#endif
  else {
    assert(false && "Unsupported memory location");
  }
}

}

}

#endif
