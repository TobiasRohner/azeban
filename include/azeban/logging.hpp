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
#ifndef LOGGING_
#define LOGGING_

#include <cstdlib>
#include <fmt/core.h>

#define AZEBAN_ERR_IF(cond, msg)                                               \
  if ((cond)) {                                                                \
    fmt::print(stderr, msg);                                                   \
    std::exit(1);                                                              \
  }

#define AZEBAN_ERR(msg)                                                        \
  fmt::print(stderr, msg);                                                     \
  std::exit(1);

#endif
