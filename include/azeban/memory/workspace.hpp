#ifndef AZEBAN_MEMORY_WORKSPACE_HPP_
#define AZEBAN_MEMORY_WORKSPACE_HPP_

#include <cstddef>
#include <zisa/memory/device_type.hpp>

namespace azeban {

class Workspace {
public:
  Workspace();
  Workspace(void *ptr, zisa::device_type loc);
  Workspace(size_t size, zisa::device_type loc = zisa::device_type::cpu);
  Workspace(const Workspace &) = delete;
  Workspace(Workspace &&) = default;
  ~Workspace();
  Workspace &operator=(const Workspace &) = delete;
  Workspace &operator=(Workspace &&);

  void *get() noexcept { return ptr_; }
  const void *get() const noexcept { return ptr_; }
  zisa::device_type location() const noexcept { return location_; }

private:
  void *ptr_;
  zisa::device_type location_;

  static void *allocate(size_t size, zisa::device_type loc);
  static void deallocate(void *ptr, zisa::device_type loc);
};

}

#endif
