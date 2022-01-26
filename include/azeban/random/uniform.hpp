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
#ifndef UNIFORM_H_
#define UNIFORM_H_

#include "random_variable.hpp"
#include <random>

namespace azeban {

template <typename Result, typename RNG>
class Uniform final : public RandomVariableImpl<Result> {
  using super = RandomVariableImpl<Result>;

public:
  using result_t = Result;

  Uniform(result_t min, result_t max, RNG &rng) : distr_(min, max), rng_(rng) {}
  Uniform(const Uniform &) = default;
  Uniform(Uniform &&) = default;

  virtual ~Uniform() override = default;

  Uniform &operator=(const Uniform &) = default;
  Uniform &operator=(Uniform &&) = default;

  virtual result_t get() override { return distr_(rng_); }

private:
  std::uniform_real_distribution<result_t> distr_;
  RNG &rng_;
};

}

#endif
