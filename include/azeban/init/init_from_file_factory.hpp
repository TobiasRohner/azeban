#ifndef INIT_FROM_FILE_FACTORY_HPP_
#define INIT_FROM_FILE_FACTORY_HPP_

#include "init_from_file.hpp"
#include <azeban/logging.hpp>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<Initializer<Dim>>
make_init_from_file(const nlohmann::json &config) {
  AZEBAN_ERR_IF(!config.contains("experiment"),
                "InitFromFile is missing parameter \"experiment\"");
  AZEBAN_ERR_IF(!config.contains("time"),
                "InitFromFile is missing parameter \"time\"");
  const std::string experiment = config["experiment"];
  const std::string time = config["time"];
  zisa::int_t sample_idx_start = 0;
  if (config.contains("sample_idx_start")) {
    sample_idx_start = config["sample_idx_start"];
  }
  return std::make_shared<InitFromFile<Dim>>(
      experiment, time, sample_idx_start);
}

}

#endif
