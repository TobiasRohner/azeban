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
#ifndef DEBUG_ARRAY_H_
#define DEBUG_ARRAY_H_

#include <azeban/memory/debug_memory_resource.hpp>
#include <zisa/memory/allocator.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>
#include <zisa/memory/memory_resource_factory.hpp>

namespace azeban {

template <typename T, int n_dims>
zisa::array<T, n_dims>
debug_array(const zisa::shape_t<n_dims> &shape,
            const std::shared_ptr<zisa::memory_resource<T>> &resource,
            const std::string &name = "Unknown") {
  const auto debug_resource
      = std::make_shared<debug_memory_resource<T>>(resource, name);
  return zisa::array<T, n_dims>(shape, zisa::allocator<T>(debug_resource));
}

template <typename T, int n_dims>
zisa::array<T, n_dims> debug_array(const zisa::shape_t<n_dims> &shape,
                                   zisa::device_type device
                                   = zisa::device_type::cpu,
                                   const std::string &name = "Unknown") {
  return debug_array<T, n_dims>(
      shape, zisa::make_memory_resource<T>(device), name);
}

}

#endif
