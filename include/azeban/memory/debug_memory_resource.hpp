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
#ifndef DEBUG_MEMORY_RESOURCE_H_
#define DEBUG_MEMORY_RESOURCE_H_

#include <fmt/core.h>
#include <string>
#include <zisa/config.hpp>
#include <zisa/memory/device_type.hpp>
#include <zisa/memory/memory_resource.hpp>

namespace azeban {

template <typename T>
class debug_memory_resource : public zisa::memory_resource<T> {
  using super = zisa::memory_resource<T>;

public:
  using value_type = typename super::value_type;
  using size_type = typename super::size_type;
  using pointer = typename super::pointer;

  explicit debug_memory_resource(
      const std::shared_ptr<zisa::memory_resource<T>> &resource,
      const std::string &name = "Unknown")
      : resource_(resource), name_(name) {}

protected:
  virtual pointer do_allocate(size_type n) override {
    pointer p = resource_->allocate(n);
    fmt::print(stderr,
               "{}: Allocarte {} bytes on {}-> {}\n",
               name_,
               n,
               location(),
               (void *)p);
    return p;
  }

  virtual void do_deallocate(pointer ptr, size_type n) {
    fmt::print(stderr,
               "{}: Deallocate {} bytes on {} at {}\n",
               name_,
               n,
               location(),
               (void *)ptr);
    resource_->deallocate(ptr, n);
  }

  virtual zisa::device_type do_device() const override {
    return resource_->device();
  }

private:
  std::shared_ptr<zisa::memory_resource<T>> resource_;
  std::string name_;

  std::string location() const {
    switch (resource_->device()) {
    case zisa::device_type::cpu:
      return "cpu";
    case zisa::device_type::cuda:
      return "cuda";
    default:
      return "unknown";
    }
  }
};

}

#endif
