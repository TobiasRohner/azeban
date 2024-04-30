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

#if AZEBAN_HAS_PYTHON

#include <azeban/init/python.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>

namespace py = pybind11;

namespace azeban {

template <int Dim>
Python<Dim>::Python(
    const std::string &script,
    std::vector<std::tuple<std::string, size_t, RandomVariable<real_t>>> &rvs)
    : script_(script), rvs_(rvs) {
  py::initialize_interpreter();
  py::exec("import sys; print(sys.path)");
}

template <int Dim>
Python<Dim>::~Python() {
  py::finalize_interpreter();
}

template <int Dim>
void Python<Dim>::do_initialize(const zisa::array_view<real_t, Dim + 1> &u) {
  const auto init = [&](auto &&u_) {
    using namespace py::literals;
    auto locals = py::dict();
    std::vector<std::tuple<std::string, std::vector<real_t>>> params;
    for (auto &[name, N, rv] : rvs_) {
      std::vector<real_t> values;
      for (size_t i = 0; i < N; ++i) {
        values.push_back(rv.get());
      }
      locals[name.c_str()] = py::array_t<real_t>(N, values.data());
    }
    std::vector<py::ssize_t> shape(Dim + 1);
    std::vector<py::ssize_t> stride(Dim + 1);
    shape[0] = u_.shape(0);
    stride[0] = sizeof(real_t) * u_.size() / u_.shape(0);
    for (int d = 1; d < Dim + 1; ++d) {
      shape[d] = u_.shape(d);
      stride[d] = stride[d - 1] / u_.shape(d);
    }
    locals["u"] = py::array_t<real_t>(shape, stride, u_.raw(), py::globals());
    py::eval_file(script_, py::globals(), locals);
  };
  if (u.memory_location() == zisa::device_type::cpu) {
    init(u);
  } else if (u.memory_location() == zisa::device_type::cuda) {
    auto h_u = zisa::array<real_t, Dim + 1>(u.shape(), zisa::device_type::cpu);
    init(h_u);
    zisa::copy(u, h_u);
  } else {
    LOG_ERR("Unknown Memory Location");
  }
}

template <int Dim>
void Python<Dim>::do_initialize(
    const zisa::array_view<complex_t, Dim + 1> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  zisa::shape_t<Dim + 1> shape;
  shape[0] = Dim;
  for (int d = 0; d < Dim; ++d) {
    shape[d + 1] = N;
  }
  auto u = zisa::array<real_t, Dim + 1>(shape, u_hat.memory_location());
  auto fft = make_fft<Dim>(u_hat, u);
  initialize(u);
  fft->forward();
}

template class Python<1>;
template class Python<2>;
template class Python<3>;

}

#endif // AZEBAN_HAS_PYTHON
