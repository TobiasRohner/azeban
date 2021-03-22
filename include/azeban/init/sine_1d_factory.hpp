#ifndef SINE_1D_FACTORY_H_
#define SINE_1D_FACTORY_H_

#include "sine_1d.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>> make_sine_1d() {
  if constexpr (Dim == 1) {
    return std::make_shared<Sine1D>();
  } else {
    fmt::print(stderr, "Sine 1D is only available for 1D simulations\n");
    exit(1);
  }
}

}

#endif
