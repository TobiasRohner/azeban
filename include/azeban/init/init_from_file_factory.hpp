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
