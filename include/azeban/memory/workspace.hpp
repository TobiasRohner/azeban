#ifndef AZEBAN_MEMORY_WORKSPACE_HPP_
#define AZEBAN_MEMORY_WORKSPACE_HPP_

#include <cstddef>
#include <zisa/memory/array_view.hpp>
#include <zisa/memory/device_type.hpp>

namespace azeban {

class Workspace {
public:
  Workspace();
  Workspace(void *ptr, zisa::device_type loc);
  Workspace(size_t size, zisa::device_type loc = zisa::device_type::cpu);
  Workspace(const Workspace &) = delete;
  Workspace(Workspace &&);
  ~Workspace();
  Workspace &operator=(const Workspace &) = delete;
  Workspace &operator=(Workspace &&);

  void *get() noexcept { return ptr_; }
  const void *get() const noexcept { return ptr_; }
  zisa::device_type location() const noexcept { return location_; }

  template <typename T, int n_dims>
  zisa::array_view<T, n_dims> get_view(size_t offset,
                                       const zisa::shape_t<n_dims> &shape) {
    LOG_ERR_IF(offset % alignof(T) != 0,
               "Given offset does not comply with alignment of T");
    uint8_t *start_ptr = static_cast<uint8_t *>(ptr_) + offset;
    return zisa::array_view<T, n_dims>(
        shape, reinterpret_cast<T *>(start_ptr), location_);
  }

  template <typename T, int n_dims>
  zisa::array_const_view<T, n_dims>
  get_view(size_t offset, const zisa::shape_t<n_dims> &shape) const {
    LOG_ERR_IF(offset % alignof(T) != 0,
               "Given offset does not comply with alignment of T");
    uint8_t *start_ptr = static_cast<uint8_t *>(ptr_) + offset;
    return zisa::array_const_view<T, n_dims>(
        shape, reinterpret_cast<T *>(start_ptr), location_);
  }

private:
  void *ptr_;
  zisa::device_type location_;

  static void *allocate(size_t size, zisa::device_type loc);
  static void deallocate(void *ptr, zisa::device_type loc);
};

}

#endif
