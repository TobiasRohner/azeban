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
#ifndef BURGERS_FACTORY_H_
#define BURGERS_FACTORY_H_

#include "burgers.hpp"

namespace azeban {

template <int Dim, typename SpectralViscosity>
std::shared_ptr<Equation<Dim>> make_burgers(const Grid<Dim> &grid,
                                            const SpectralViscosity &visc,
                                            zisa::device_type device) {
  if constexpr (Dim == 1) {
    return std::make_shared<Burgers<SpectralViscosity>>(grid, visc, device);
  } else {
    ZISA_UNUSED(grid);
    ZISA_UNUSED(visc);
    ZISA_UNUSED(device);
    fmt::print(stderr, "Burgers is only implemented for 1D\n");
    exit(1);
  }
}

}

#endif
