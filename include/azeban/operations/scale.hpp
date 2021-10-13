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
#ifndef SCALE_H_
#define SCALE_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/operations/scale_cuda.hpp>
#endif

namespace azeban {

template <typename Scalar, int Dim>
void scale(const Scalar &a, const zisa::array_view<Scalar, Dim> &x) {
  const zisa::shape_t<1> flat_shape{zisa::product(x.shape())};
  const zisa::array_view<Scalar, 1> x_flat(
      flat_shape, x.raw(), x.memory_location());
  if (x_flat.memory_location() == zisa::device_type::cpu) {
    for (zisa::int_t i = 0; i < x_flat.shape(0); ++i) {
      x_flat[i] *= a;
    }
  }
#if ZISA_HAS_CUDA
  else if (x_flat.memory_location() == zisa::device_type::cuda) {
    scale_cuda<Scalar>(a, x_flat);
  }
#endif
  else {
    LOG_ERR("Unsupported Memory Location");
  }
}

}

#endif
