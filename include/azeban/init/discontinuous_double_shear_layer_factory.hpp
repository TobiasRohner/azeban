#ifndef DISCONTINUOUS_DOUBLE_SHEAR_LAYER_FACTORY_H_
#define DISCONTINUOUS_DOUBLE_SHEAR_LAYER_FACTORY_H_

#include "discontinuous_double_shear_layer.hpp"
#include "init_3d_from_2d.hpp"
#include <azeban/logging.hpp>
#include <azeban/random/random_variable_factory.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>>
make_discontinuous_double_shear_layer(const nlohmann::json &config, RNG &rng) {
  if constexpr (Dim == 2 || Dim == 3) {
    AZEBAN_ERR_IF(!config.contains("delta"),
                  "Discontinuous Double Shear Layer initialization is missing "
                  "parameter \"delta\"\n");

    RandomVariable<real_t> delta
        = make_random_variable<real_t>(config["delta"], rng);
    if constexpr (Dim == 2) {
      return std::make_shared<DiscontinuousDoubleShearLayer>(delta);
    } else {
      AZEBAN_ERR_IF(
          !config.contains("dimension"),
          "Must specify constant \"dimension\" to generalize from 2D to 3D\n");

      const int dim = config["dimension"];
      auto init2d = std::make_shared<DiscontinuousDoubleShearLayer>(delta);
      return std::make_shared<Init3DFrom2D>(dim, init2d);
    }
  }

  AZEBAN_ERR("Discontinuous Double Shear Layer is only available for 2D or 3D "
             "simulations\n");
}

}

#endif
