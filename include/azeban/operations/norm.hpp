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
#ifndef NORM_H_
#define NORM_H_

#include <azeban/profiler.hpp>
#include <zisa/config.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/norm_cuda.hpp>
#endif

namespace azeban {

template <int Dim, typename Scalar>
real_t norm(const zisa::array_const_view<Scalar, Dim> &data, real_t p) {
  ProfileHost profile("norm");
  if (data.memory_location() == zisa::device_type::cpu) {
    real_t val = 0;
    for (zisa::int_t i = 0; i < zisa::product(data.shape()); ++i) {
      using zisa::abs;
      val += zisa::pow(abs(data[i]), p);
    }
    return zisa::pow(val, real_t(1. / p));
  }
#if ZISA_HAS_CUDA
  else if (data.memory_location() == zisa::device_type::cuda) {
    zisa::array_const_view<Scalar, 1> view(
        zisa::shape_t<1>(zisa::product(data.shape())),
        data.raw(),
        data.memory_location());
    return norm_cuda(view, p);
  }
#endif
  else {
    LOG_ERR("Unsupported memory location");
  }
  // Make compiler happy
  return 0;
}

template <int Dim, typename Scalar>
real_t norm(const zisa::array_view<Scalar, Dim> &data, real_t p) {
  return norm(zisa::array_const_view<Scalar, Dim>(
                  data.shape(), data.raw(), data.memory_location()),
              p);
}

template <int Dim, typename Scalar>
real_t norm(const zisa::array<Scalar, Dim> &data, real_t p) {
  return norm(zisa::array_const_view<Scalar, Dim>(
                  data.shape(), data.raw(), data.device()),
              p);
}

template <int Dim, typename Scalar>
real_t max_norm(const zisa::array_const_view<Scalar, Dim> &data) {
  ProfileHost profile("max_norm");
  if (data.memory_location() == zisa::device_type::cpu) {
    using zisa::abs;
    real_t val = abs(data[0]);
    for (zisa::int_t i = 0; i < zisa::product(data.shape()); ++i) {
      using zisa::abs;
      val = zisa::max(val, real_t(abs(data[i])));
    }
    return val;
  }
#if ZISA_HAS_CUDA
  else if (data.memory_location() == zisa::device_type::cuda) {
    zisa::array_const_view<Scalar, 1> view(
        zisa::shape_t<1>(zisa::product(data.shape())),
        data.raw(),
        data.memory_location());
    return max_norm_cuda(view);
  }
#endif
  else {
    LOG_ERR("Unsupported memory location");
  }
  // Make compiler happy
  return 0;
}

template <int Dim, typename Scalar>
real_t max_norm(const zisa::array_view<Scalar, Dim> &data) {
  return max_norm(zisa::array_const_view<Scalar, Dim>(
      data.shape(), data.raw(), data.memory_location()));
}

template <int Dim, typename Scalar>
real_t max_norm(const zisa::array<Scalar, Dim> &data) {
  return max_norm(zisa::array_const_view<Scalar, Dim>(
      data.shape(), data.raw(), data.device()));
}

}

#endif
