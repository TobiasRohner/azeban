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
