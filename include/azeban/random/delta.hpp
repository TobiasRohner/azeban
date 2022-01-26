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
#ifndef DELTA_H_
#define DELTA_H_

#include "random_variable.hpp"

namespace azeban {

template <typename Result>
class Delta final : public RandomVariableImpl<Result> {
  using super = RandomVariableImpl<Result>;

public:
  using result_t = Result;

  Delta(result_t result) : result_(result) {}
  Delta(const Delta &) = default;
  Delta(Delta &&) = default;

  virtual ~Delta() override = default;

  Delta &operator=(const Delta &) = default;
  Delta &operator=(Delta &&) = default;

  virtual result_t get() override { return result_; }

private:
  result_t result_;
};

}

#endif
