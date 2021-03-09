#ifndef GRID_FACTORY_H_
#define GRID_FACTORY_H_

#include <azeban/grid.hpp>
#include <fmt/core.h>

namespace azeban {

template <int Dim, typename Json>
Grid<Dim> make_grid(Json &&config) {
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
      grid.N_phys_pad = config["N_phys_pad"];
      grid.N_fourier_pad = grid.N_phys_pad / 2 + 1;
    } else if (config.contains("N_fourier_pad")
               && !config.contains("N_phys_pad")) {
      grid.N_fourier_pad = config["N_fourier_pad"];
      grid.N_phys_pad = 2 * (grid.N_fourier_pad - 1);
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
