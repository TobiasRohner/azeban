#ifndef TAYLOR_GREEN_FACTORY_H_
#define TAYLOR_GREEN_FACTORY_H_

#include "taylor_green.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>> make_taylor_green() {
  if constexpr (Dim == 2 || Dim == 3) {
    return std::make_shared<TaylorGreen<Dim>>();
  } else {
    fmt::print(stderr,
               "Taylor Green  is only available for 2D or 3D simulations\n");
    exit(1);
  }
}

}

#endif
