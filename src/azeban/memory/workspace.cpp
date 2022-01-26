#include <azeban/logging.hpp>
#include <azeban/memory/workspace.hpp>
#if ZISA_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace azeban {

Workspace::Workspace() : Workspace(nullptr, zisa::device_type::unknown) {}

Workspace::Workspace(void *ptr, zisa::device_type loc)
    : ptr_(ptr), location_(loc) {}

Workspace::Workspace(size_t size, zisa::device_type loc)
    : Workspace(allocate(size, loc), loc) {}

Workspace::~Workspace() {
  if (ptr_) {
    deallocate(ptr_, location_);
  }
}

Workspace &Workspace::operator=(Workspace &&other) {
  if (&other == this) {
    return *this;
  }
  if (ptr_) {
    deallocate(ptr_, location_);
  }
  ptr_ = other.ptr_;
  location_ = other.location_;
  return *this;
}

void *Workspace::allocate(size_t size, zisa::device_type loc) {
  if (loc == zisa::device_type::cpu) {
    return malloc(size);
  }
#if ZISA_HAS_CUDA
  else if (loc == zisa::device_type::cuda) {
    void *ptr;
    cudaMalloc(&ptr, size);
    return ptr;
  }
#endif
  else {
    AZEBAN_ERR("Unknown memory location\n");
  }
}

void Workspace::deallocate(void *ptr, zisa::device_type loc) {
  if (loc == zisa::device_type::cpu) {
    free(ptr);
  }
#if ZISA_HAS_CUDA
  else if (loc == zisa::device_type::cuda) {
    cudaFree(ptr);
  }
#endif
  else {
    AZEBAN_ERR("Unknown memory location\n");
  }
}

}
