#ifndef AZEBAN_MEMORY_PINNED_ARRAY_HPP_
#define AZEBAN_MEMORY_PINNED_ARRAY_HPP_

#include <azeban/memory/pinned_memory_resource.hpp>
#include <zisa/memory/allocator.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

template <typename T, int n_dims>
zisa::array<T, n_dims> pinned_array(const zisa::shape_t<n_dims> &shape) {
  const auto pinned_resource = std::make_shared<pinned_memory_resource<T>>();
  return zisa::array<T, n_dims>(shape, zisa::allocator<T>(pinned_resource));
}

}

#endif
