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
#ifndef TEST_AZEBAN_UTILS_HPP
#define TEST_AZEBAN_UTILS_HPP

#include <azeban/config.hpp>
#include <azeban/operations/copy_to_padded.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>

template <typename T, int D>
zisa::array_view<T, D - 1> component(const zisa::array_view<T, D> &arr,
                                     zisa::int_t n) {
  zisa::shape_t<D - 1> slice_shape;
  for (zisa::int_t i = 0; i < D - 1; ++i) {
    slice_shape[i] = arr.shape(i + 1);
  }
  return zisa::array_view<T, D - 1>(slice_shape,
                                    arr.raw() + n * zisa::product(slice_shape),
                                    arr.memory_location());
}

template <typename T, int D>
zisa::array_const_view<T, D - 1>
component(const zisa::array_const_view<T, D> &arr, zisa::int_t n) {
  zisa::shape_t<D - 1> slice_shape;
  for (zisa::int_t i = 0; i < D - 1; ++i) {
    slice_shape[i] = arr.shape(i + 1);
  }
  return zisa::array_const_view<T, D - 1>(slice_shape,
                                          arr.raw()
                                              + n * zisa::product(slice_shape),
                                          arr.memory_location());
}

template <typename T, int D>
zisa::array_view<T, D - 1> component(zisa::array<T, D> &arr, zisa::int_t n) {
  zisa::shape_t<D - 1> slice_shape;
  for (zisa::int_t i = 0; i < D - 1; ++i) {
    slice_shape[i] = arr.shape(i + 1);
  }
  return zisa::array_view<T, D - 1>(
      slice_shape, arr.raw() + n * zisa::product(slice_shape), arr.device());
}

template <zisa::int_t Dim>
azeban::real_t
L2(const zisa::array_const_view<azeban::complex_t, Dim + 1> &u,
   const zisa::array_const_view<azeban::complex_t, Dim + 1> &u_ref) {
  static_assert(
      Dim >= 2,
      "L2 error only supported for dimensions strictly larger than 1");

  LOG_ERR_IF(u.memory_location() != zisa::device_type::cpu,
             "u must be in host memory");
  LOG_ERR_IF(u_ref.memory_location() != zisa::device_type::cpu,
             "u_ref must be in host memory");
  LOG_ERR_IF(u.shape(0) != u_ref.shape(0), "Mismatch in number of components");

  const zisa::int_t N = u.shape(1);
  const zisa::int_t N_ref = u_ref.shape(1);

  zisa::array<azeban::complex_t, Dim + 1> u_pad(u_ref.shape());
  for (zisa::int_t d = 0; d < u.shape(0); ++d) {
    azeban::copy_to_padded(component(u_pad, d), component(u, d), 0);
  }
  for (zisa::int_t i = 0; i < u_pad.size(); ++i) {
    u_pad[i] *= zisa::pow<Dim>(static_cast<azeban::real_t>(N_ref) / N);
  }

  azeban::real_t errL2 = 0;
  if constexpr (Dim == 2) {
    for (zisa::int_t d = 0; d < u_ref.shape(0); ++d) {
      for (zisa::int_t i = 0; i < u_ref.shape(1); ++i) {
        errL2 += azeban::abs2(u_pad(d, i, 0) - u_ref(d, i, 0));
        for (zisa::int_t j = 1; j < u_ref.shape(2); ++j) {
          errL2 += 2 * azeban::abs2(u_pad(d, i, j) - u_ref(d, i, j));
        }
      }
    }
  } else if constexpr (Dim == 3) {
    for (zisa::int_t d = 0; d < u_ref.shape(0); ++d) {
      for (zisa::int_t i = 0; i < u_ref.shape(1); ++i) {
        for (zisa::int_t j = 0; j < u_ref.shape(2); ++j) {
          errL2 += azeban::abs2(u_pad(d, i, j, 0) - u_ref(d, i, j, 0));
          for (zisa::int_t k = 1; k < u_ref.shape(3); ++k) {
            errL2 += 2 * azeban::abs2(u_pad(d, i, j, k) - u_ref(d, i, j, k));
          }
        }
      }
    }
  }
  errL2 = zisa::sqrt(errL2) / zisa::pow<Dim>(N_ref);

  return errL2;
}

#endif
