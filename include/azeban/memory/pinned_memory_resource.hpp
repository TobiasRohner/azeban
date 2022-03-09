#ifndef AZEBAN_MEMORY_PINNED_MEMORY_RESOURCE_HPP_
#define AZEBAN_MEMORY_PINNED_MEMORY_RESOURCE_HPP_

#include <azeban/cuda/cuda_check_error.hpp>
#include <zisa/memory/memory_resource.hpp>

namespace azeban {

template <typename T>
class pinned_memory_resource : public zisa::memory_resource<T> {
  using super = zisa::memory_resource<T>;

public:
  using value_type = typename super::value_type;
  using size_type = typename super::size_type;
  using pointer = typename super::pointer;

protected:
  virtual pointer do_allocate(size_type n) override {
    pointer p;
    const auto err = cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocDefault);
    cudaCheckError(err);
    return p;
  }

  virtual void do_deallocate(pointer ptr, size_type n) override {
    const auto err = cudaFreeHost(ptr);
    cudaCheckError(err);
  }

  virtual zisa::device_type do_device() const override {
    return zisa::device_type::cpu;
  }
};

}

#endif
