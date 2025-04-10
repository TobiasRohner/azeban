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
#ifndef CLAMP_H_
#define CLAMP_H_

#include <azeban/config.hpp>
#include <cmath>
#include <zisa/memory/array_view.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/operations/clamp_cuda.hpp>
#endif

namespace azeban {

template <typename Scalar, int Dim>
void clamp(const zisa::array_view<Scalar, Dim> &x, real_t a) {
  const zisa::shape_t<1> flat_shape{zisa::product(x.shape())};
  const zisa::array_view<Scalar, 1> x_flat(
      flat_shape, x.raw(), x.memory_location());
  if (x_flat.memory_location() == zisa::device_type::cpu) {
    for (zisa::int_t i = 0; i < x_flat.shape(0); ++i) {
      Scalar value = x_flat[i];
      x_flat[i] = value / zisa::max(real_t{1}, abs(value) / a);
    }
  }
#if ZISA_HAS_CUDA
  else if (x_flat.memory_location() == zisa::device_type::cuda) {
    clamp_cuda<Scalar>(x_flat, a);
  }
#endif
  else {
    LOG_ERR("Unsupported Memory Location");
  }
}

}

#endif
