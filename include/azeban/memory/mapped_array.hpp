#ifndef AZEBAN_MEMORY_MAPPED_ARRAY_HPP_
#define AZEBAN_MEMORY_MAPPED_ARRAY_HPP_

#include <azeban/memory/mapped_memory_resource.hpp>
#include <zisa/memory/allocator.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>
#include <zisa/memory/memory_resource_factory.hpp>


namespace azeban {

template<typename T, int n_dims>
zisa::array<T, n_dims> mapped_array(const zisa::shape_t<n_dims> &shape) {
  const auto mapped_resource = std::make_shared<mapped_memory_resource<T>>();
  return zisa::array<T, n_dims>(shape, zisa::allocator<T>(mapped_resource));
}

template<typename T, int n_dims>
zisa::array_view<T, n_dims> mapped_array_to_cuda_view(zisa::array<T, n_dims> &arr) {
  T *ptr;
  cudaHostGetDevicePointer(&ptr, arr.raw(), 0);
  return zisa::array_view<T, n_dims>(arr.shape(), ptr, zisa::device_type::cuda);
}

template<typename T, int n_dims>
zisa::array_const_view<T, n_dims> mapped_array_to_cuda_view(const zisa::array<T, n_dims> &arr) {
  T *ptr;
  cudaHostGetDevicePointer(&ptr, arr.raw(), 0);
  return zisa::array_const_view<T, n_dims>(arr.shape(), ptr, zisa::device_type::cuda);
}

}


#endif
