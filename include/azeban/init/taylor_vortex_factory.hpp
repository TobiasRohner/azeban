#ifndef TAYLOR_VORTEX_FACTORY_H_
#define TAYLOR_VORTEX_FACTORY_H_

#include "init_3d_from_2d.hpp"
#include "taylor_vortex.hpp"
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>>
make_taylor_vortex(const nlohmann::json &config) {
  if constexpr (Dim == 2) {
    return std::make_shared<TaylorVortex>();
  } else if constexpr (Dim == 3) {
    if (!config.contains("dimension")) {
      fmt::print(stderr,
                 "Must specify constant \"dimension\" to generalize from 2D "
                 "to 3D\n");
      exit(1);
    }
    const int dim = config["dimension"];
    const auto init2d = std::make_shared<TaylorVortex>();
    return std::make_shared<Init3DFrom2D>(dim, init2d);
  } else {
    fmt::print(stderr,
               "Taylor Vortex is only available for 2D or 3D simulations\n");
    exit(1);
  }
}

}

#endif
