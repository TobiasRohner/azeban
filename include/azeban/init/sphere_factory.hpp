#ifndef SPHERE_FACTORY_H_
#define SPHERE_FACTORY_H_

#include "sphere.hpp"
#include <fmt/core.h>
#include <vector>

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>> make_sphere(const nlohmann::json &config) {
  if constexpr (Dim == 2 || Dim == 3) {
    if (!config.contains("center")) {
      fmt::print(stderr, "Sphere initializer needs parameter \"center\"\n");
      exit(1);
    }
    if (!config.contains("radius")) {
      fmt::print(stderr, "Sphere initializer needs parameter \"radius\"\n");
      exit(1);
    }
    const std::vector<real_t> center
        = config["center"].get<std::vector<real_t>>();
    const real_t radius = config["radius"];
    if (center.size() != Dim) {
      fmt::print(stderr,
                 "Center of sphere initializer must have {} components instead "
                 "of {}\n",
                 Dim,
                 center.size());
      exit(1);
    }
    if constexpr (Dim == 2) {
      return std::make_shared<Sphere<Dim>>(
          std::array<real_t, 2>{center[0], center[1]}, radius);
    } else {
      return std::make_shared<Sphere<Dim>>(
          std::array<real_t, 3>{center[0], center[1], center[2]}, radius);
    }
  } else {
    fmt::print(stderr, "Sphere initializer is only defined for 2D or 3D\n");
    exit(1);
  }
}

}

#endif
