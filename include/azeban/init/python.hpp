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
#ifndef AZEBAN_INIT_PYTHON_HPP
#define AZEBAN_INIT_PYTHON_HPP

#include "initializer.hpp"
#include <azeban/random/random_variable.hpp>
#include <utility>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
class Python final : public Initializer<Dim> {
  using super = Initializer<Dim>;

public:
  Python(const std::string &script,
         std::vector<std::tuple<std::string, size_t, RandomVariable<real_t>>>
             &rvs);
  Python(const Python &) = default;
  Python(Python &&) = default;

  ~Python();

  Python &operator=(const Python &) = default;
  Python &operator=(Python &&) = default;

  using super::initialize;

protected:
  virtual void
  do_initialize(const zisa::array_view<real_t, Dim + 1> &u) override;
  virtual void
  do_initialize(const zisa::array_view<complex_t, Dim + 1> &u_hat) override;

private:
  std::string script_;
  std::vector<std::tuple<std::string, size_t, RandomVariable<real_t>>> rvs_;
};

}

#endif
