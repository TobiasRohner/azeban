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
#ifndef GRID_FACTORY_H_
#define GRID_FACTORY_H_

#include <azeban/grid.hpp>
#include <azeban/operations/fft.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
Grid<Dim> make_grid(const nlohmann::json &config, zisa::device_type device) {
  Grid<Dim> grid;
  if (config.contains("N_phys") && !config.contains("N_fourier")) {
    grid.N_phys = config["N_phys"];
    grid.N_fourier = grid.N_phys / 2 + 1;
    grid.N_phys_pad = 3 * grid.N_phys / 2;
    grid.N_fourier_pad = grid.N_phys_pad / 2 + 1;
  } else if (config.contains("N_fourier") && !config.contains("N_phys")) {
    grid.N_fourier = config["N_fourier"];
    grid.N_phys = 2 * (grid.N_fourier - 1);
    grid.N_phys_pad = 3 * grid.N_phys / 2;
    grid.N_fourier_pad = grid.N_phys_pad / 2 + 1;
  } else {
    fmt::print(stderr,
               "Grid config must contain either \"N_phys\" or \"N_fourier\", "
               "but not both\n");
    exit(1);
  }
  if (config.contains("N_phys_pad") || config.contains("N_fourier_pad")) {
    if (config.contains("N_phys_pad") && !config.contains("N_fourier_pad")) {
      if (config["N_phys_pad"].is_string()) {
        grid.N_phys_pad
            = optimal_fft_size(config["N_phys_pad"],
                               zisa::div_up(3 * grid.N_phys, zisa::int_t(2)),
                               Dim,
                               Dim,
                               device);
        fmt::print(stderr,
                   "Info: Minimal padding size given from \"N_phys_pad\" is "
                   "{}. Padded to {} for speed.\n",
                   zisa::div_up(3 * grid.N_phys, zisa::int_t(2)),
                   grid.N_phys_pad);
      } else {
        grid.N_phys_pad = config["N_phys_pad"];
      }
      grid.N_fourier_pad = grid.N_phys_pad / 2 + 1;
    } else if (config.contains("N_fourier_pad")
               && !config.contains("N_phys_pad")) {
      if (config["N_fourier_pad"].is_string()) {
        grid.N_phys_pad
            = optimal_fft_size(config["N_phys_pad"],
                               zisa::div_up(3 * grid.N_phys, zisa::int_t(2)),
                               Dim,
                               Dim,
                               device);
        fmt::print(stderr,
                   "Info: Minimal padding size given from \"N_fourier_pad\" is "
                   "{}. Padded to {} for speed.\n",
                   zisa::div_up(3 * grid.N_phys, zisa::int_t(2)),
                   grid.N_phys_pad);
        grid.N_fourier_pad = grid.N_phys_pad / 2 + 1;
      } else {
        grid.N_fourier_pad = config["N_fourier_pad"];
        grid.N_phys_pad = 2 * (grid.N_fourier_pad - 1);
      }
    } else {
      fmt::print(stderr,
                 "Grid config may contain \"N_phys_pad\" or \"N_fourier_pad\", "
                 "bot not both\n");
      exit(1);
    }
  }
  return grid;
}

}

#endif
