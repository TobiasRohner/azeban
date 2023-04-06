#ifndef AZEBAN_INIT_PYTHON_FACTORY_HPP_
#define AZEBAN_INIT_PYTHON_FACTORY_HPP_

#include "python.hpp"
#include <azeban/random/random_variable_factory.hpp>
#include <nlohmann/json.hpp>

namespace azeban {

template <typename RNG>
std::tuple<std::string, size_t, RandomVariable<real_t>>
python_make_param(const nlohmann::json &config, RNG &rng) {
  AZEBAN_ERR_IF(!config.contains("name"),
                "Python Initializer Parameter is missing \"name\"\n");
  AZEBAN_ERR_IF(!config.contains("value"),
                "Python Initializer Parameter is missing \"value\"\n");

  const std::string name = config["name"];
  size_t N = 1;
  if (config.contains("N")) {
    N = config["N"];
  }
  RandomVariable<real_t> value
      = make_random_variable<real_t>(config["value"], rng);

  return std::make_tuple(name, N, value);
}

template <typename RNG>
std::vector<std::tuple<std::string, size_t, RandomVariable<real_t>>>
python_make_params(const nlohmann::json &config, RNG &rng) {
  std::vector<std::tuple<std::string, size_t, RandomVariable<real_t>>> params;
  if (config.is_array()) {
    for (const auto &p : config) {
      params.push_back(python_make_param(p, rng));
    }
  } else {
    params.push_back(python_make_param(config, rng));
  }
  return params;
}

template <int Dim, typename RNG>
std::shared_ptr<Initializer<Dim>> make_python(const nlohmann::json &config,
                                              RNG &rng) {
  AZEBAN_ERR_IF(!config.contains("script"),
                "Python Initializer is missing \"script\" parameter\n");
  const std::string script = config["script"];
  std::vector<std::tuple<std::string, size_t, RandomVariable<real_t>>> params;
  if (config.contains("params")) {
    params = python_make_params(config["params"], rng);
  }
  return std::make_shared<Python<Dim>>(script, params);
}

}

#endif
