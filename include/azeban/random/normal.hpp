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
#ifndef NORMAL_H_
#define NORMAL_H_

#include "random_variable.hpp"
#include <random>

namespace azeban {

template <typename Result, typename RNG>
class Normal final : public RandomVariableImpl<Result> {
  using super = RandomVariableImpl<Result>;

public:
  using result_t = Result;

  Normal(result_t mu, result_t sigma, RNG &rng)
      : distr_(mu, sigma), rng_(rng) {}
  Normal(const Normal &) = default;
  Normal(Normal &&) = default;

  virtual ~Normal() override = default;

  Normal &operator=(const Normal &) = default;
  Normal &operator=(Normal &&) = default;

  virtual result_t get() override { return distr_(rng_); }

private:
  std::normal_distribution<result_t> distr_;
  RNG &rng_;
};

}

#endif
