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
#ifndef RANDOM_VARIABLE_H_
#define RANDOM_VARIABLE_H_

#include <memory>

namespace azeban {

template <typename Result>
class RandomVariableImpl {
public:
  using result_t = Result;

  RandomVariableImpl() = default;
  RandomVariableImpl(const RandomVariableImpl &) = default;
  RandomVariableImpl(RandomVariableImpl &&) = default;

  virtual ~RandomVariableImpl() = default;

  RandomVariableImpl &operator=(const RandomVariableImpl &) = default;
  RandomVariableImpl &operator=(RandomVariableImpl &&) = default;

  virtual Result get() = 0;
};

template <typename Result>
class RandomVariable final {
public:
  using result_t = Result;

  RandomVariable(const std::shared_ptr<RandomVariableImpl<Result>> &rv)
      : rv_(rv) {}
  RandomVariable(const RandomVariable &) = default;
  RandomVariable(RandomVariable &&) = default;

  ~RandomVariable() = default;

  RandomVariable &operator=(const RandomVariable &) = default;
  RandomVariable &operator=(RandomVariable &&) = default;

  operator result_t() { return get(); }

  Result get() { return rv_->get(); }

private:
  std::shared_ptr<RandomVariableImpl<Result>> rv_;
};

}

#endif
