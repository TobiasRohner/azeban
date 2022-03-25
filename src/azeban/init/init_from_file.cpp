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
#include <azeban/init/init_from_file.hpp>
#include <azeban/logging.hpp>
#include <azeban/operations/fft.hpp>
#include <azeban/operations/inverse_curl.hpp>
#include <zisa/io/netcdf_serial_writer.hpp>

namespace azeban {

template <int Dim>
void InitFromFile<Dim>::do_initialize(
    const zisa::array_view<real_t, Dim + 1> &u) {
  int status, ncid;
  status = nc_open(filename().c_str(), 0, &ncid);
  AZEBAN_ERR_IF(status != NC_NOERR, "Failed to open initial conditions");
  if (u.memory_location() == zisa::device_type::cpu) {
    init(ncid, u);
  } else if (u.memory_location() == zisa::device_type::cuda) {
    auto h_u = zisa::array<real_t, Dim + 1>(u.shape(), zisa::device_type::cpu);
    init(ncid, h_u);
    zisa::copy(u, h_u);
  } else {
    AZEBAN_ERR("Unknown memory location");
  }
  ++sample_;
  status = nc_close(ncid);
  AZEBAN_ERR_IF(status != NC_NOERR, "Failed to close file");
}

template <int Dim>
void InitFromFile<Dim>::do_initialize(
    const zisa::array_view<complex_t, Dim + 1> &u_hat) {
  int status, ncid;
  status = nc_open(filename().c_str(), 0, &ncid);
  AZEBAN_ERR_IF(status != NC_NOERR, "Failed to open initial conditions");
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    init(ncid, u_hat);
  } else if (u_hat.memory_location() == zisa::device_type::cuda) {
    auto h_u_hat = zisa::array<complex_t, Dim + 1>(u_hat.shape(),
                                                   zisa::device_type::cpu);
    init(ncid, h_u_hat);
    zisa::copy(u_hat, h_u_hat);
  } else {
    AZEBAN_ERR("Unknown memory location");
  }
  ++sample_;
  status = nc_close(ncid);
  AZEBAN_ERR_IF(status != NC_NOERR, "Failed to close file");
}

template <int Dim>
std::string InitFromFile<Dim>::filename() const {
  return experiment_ + "/sample_" + std::to_string(sample_) + "_time_" + time_
         + ".nc";
}

template <int Dim>
std::vector<int> InitFromFile<Dim>::get_varids(int ncid) const {
  int status;
  int nvars;
  status = nc_inq_nvars(ncid, &nvars);
  AZEBAN_ERR_IF(status != NC_NOERR, "Unable to get number of variables");
  std::vector<int> varids(nvars);
  status = nc_inq_varids(ncid, &nvars, varids.data());
  AZEBAN_ERR_IF(status != NC_NOERR, "Unable to get variable IDs");
  varids.resize(nvars);
  return varids;
}

template <int Dim>
std::vector<std::string> InitFromFile<Dim>::get_varnames(int ncid) const {
  int status;
  std::vector<std::string> varnames;
  const std::vector<int> varids = get_varids(ncid);
  char name[NC_MAX_NAME];
  for (int varid : varids) {
    status = nc_inq_varname(ncid, varid, name);
    AZEBAN_ERR_IF(status != NC_NOERR, "Unable to get variable name");
    varnames.emplace_back(name);
  }
  return varnames;
}

template <int Dim>
std::vector<size_t> InitFromFile<Dim>::get_dims(int ncid, int varid) const {
  int status;
  int ndims;
  int dimids[3];
  status = nc_inq_varndims(ncid, varid, &ndims);
  AZEBAN_ERR_IF(status != NC_NOERR, "Unable to get number of dimensions");
  status = nc_inq_vardimid(ncid, varid, dimids);
  AZEBAN_ERR_IF(status != NC_NOERR, "Unable to read dimension IDs");
  std::vector<size_t> dims(ndims);
  for (int i = 0; i < ndims; ++i) {
    status = nc_inq_dimlen(ncid, dimids[i], &(dims[i]));
    AZEBAN_ERR_IF(status != NC_NOERR, "Unable to read dimension length");
  }
  return dims;
}

template <int Dim>
void InitFromFile<Dim>::read_component(
    int ncid,
    const std::string &name,
    const zisa::array_view<real_t, Dim> &u) const {
  int status, varid;
  status = nc_inq_varid(ncid, name.c_str(), &varid);
  AZEBAN_ERR_IF(status != NC_NOERR, "File does not contain variable");
  const auto dims = get_dims(ncid, varid);
  AZEBAN_ERR_IF(dims.size() != Dim, "Initial data has wrong dimension");
  for (int i = 0; i < Dim; ++i) {
    AZEBAN_ERR_IF(dims[i] != u.shape(i), "Given array has wrong shape");
  }
  status = nc_get_var(ncid, varid, u.raw());
  AZEBAN_ERR_IF(status != NC_NOERR, "Could not read variable");
}

template <int Dim>
void InitFromFile<Dim>::read_u(
    int ncid, const zisa::array_view<real_t, Dim + 1> &u) const {
  AZEBAN_ERR_IF(u.memory_location() != zisa::device_type::cpu,
                "Expected CPU array");
  zisa::shape_t<Dim> view_shape;
  for (int d = 0; d < Dim; ++d) {
    view_shape[d] = u.shape(d + 1);
  }
  const zisa::int_t view_size = zisa::product(view_shape);
  zisa::array_view<real_t, Dim> view_u(
      view_shape, u.raw(), zisa::device_type::cpu);
  read_component(ncid, "u", view_u);
  zisa::array_view<real_t, Dim> view_v(
      view_shape, u.raw() + view_size, zisa::device_type::cpu);
  read_component(ncid, "v", view_v);
  if (Dim > 2) {
    zisa::array_view<real_t, Dim> view_w(
        view_shape, u.raw() + 2 * view_size, zisa::device_type::cpu);
    read_component(ncid, "w", view_w);
  }
}

template <int Dim>
void InitFromFile<Dim>::read_u_hat(
    int ncid, const zisa::array_view<complex_t, Dim + 1> &u_hat) const {
  const zisa::int_t N_phys = u_hat.shape(1);
  zisa::shape_t<Dim + 1> shape_u;
  shape_u[0] = Dim;
  for (int d = 0; d < Dim; ++d) {
    shape_u[d + 1] = N_phys;
  }
  zisa::array<real_t, Dim + 1> u(shape_u, zisa::device_type::cpu);
  auto fft = make_fft<Dim>(u_hat, u);
  read_u(ncid, u);
  fft->forward();
}

template <int Dim>
void InitFromFile<Dim>::read_omega(
    int ncid, const zisa::array_view<real_t, Dim + 1> &omega) const {
  AZEBAN_ERR_IF(Dim != 2, "Loading from vorticity only supported for 2D");
  const zisa::int_t N_phys = omega.shape(1);
  zisa::shape_t<Dim> shape_omega_view;
  for (int d = 0; d < Dim; ++d) {
    shape_omega_view[d] = N_phys;
  }
  zisa::array_view<real_t, Dim> omega_view(
      shape_omega_view, omega.raw(), omega.memory_location());
  read_component(ncid, "omega", omega_view);
}

template <int Dim>
void InitFromFile<Dim>::read_omega_hat(
    int ncid, const zisa::array_view<complex_t, Dim + 1> &omega_hat) const {
  AZEBAN_ERR_IF(Dim != 2, "Loading from vorticity only supported for 2D");
  const zisa::int_t N_phys = omega_hat.shape(1);
  zisa::shape_t<Dim + 1> shape_omega;
  shape_omega[0] = 1;
  for (int d = 0; d < Dim; ++d) {
    shape_omega[d + 1] = N_phys;
  }
  zisa::array<real_t, Dim + 1> omega(shape_omega, zisa::device_type::cpu);
  auto fft = make_fft<Dim>(omega_hat, omega);
  read_omega(ncid, omega);
  fft->forward();
}

template <int Dim>
void InitFromFile<Dim>::init(int ncid,
                             const zisa::array_view<real_t, Dim + 1> &u) const {
  const auto varnames = get_varnames(ncid);
  bool contains_u = false;
  bool contains_v = false;
  bool contains_w = false;
  bool contains_omega = false;
  for (const auto &varname : varnames) {
    if (varname == "u") {
      contains_u = true;
    }
    if (varname == "v") {
      contains_v = true;
    }
    if (varname == "w") {
      contains_w = true;
    }
    if (varname == "omega") {
      contains_omega = true;
    }
  }
  if (contains_u && (Dim <= 1 || contains_v) && (Dim <= 2 || contains_w)) {
    read_u(ncid, u);
  } else if (Dim == 2 && contains_omega) {
    const zisa::int_t N_phys = u.shape(1);
    zisa::shape_t<Dim + 1> shape_omega_hat;
    zisa::shape_t<Dim + 1> shape_u_hat;
    shape_omega_hat[0] = 1;
    shape_u_hat[0] = u.shape(0);
    for (int d = 0; d < Dim - 1; ++d) {
      shape_omega_hat[d + 1] = N_phys;
      shape_u_hat[d + 1] = N_phys;
    }
    shape_omega_hat[Dim] = N_phys / 2 + 1;
    shape_u_hat[Dim] = N_phys / 2 + 1;
    zisa::array<complex_t, Dim + 1> omega_hat(shape_omega_hat,
                                              zisa::device_type::cpu);
    zisa::array<complex_t, Dim + 1> u_hat(shape_u_hat, zisa::device_type::cpu);
    auto fft = make_fft<Dim>(u_hat, u);
    read_omega_hat(ncid, omega_hat);
    zisa::shape_t<2> omega_hat_view_shape{N_phys, N_phys / 2 + 1};
    zisa::array_view<complex_t, 2> omega_hat_view(
        omega_hat_view_shape, omega_hat.raw(), omega_hat.device());
    zisa::shape_t<3> u_hat_view_shape{2, N_phys, N_phys / 2 + 1};
    zisa::array_view<complex_t, 3> u_hat_view(
        u_hat_view_shape, u_hat.raw(), u_hat.device());
    inverse_curl(omega_hat_view, u_hat_view);
    fft->backward();
  } else {
    LOG_ERR("Invlid input file provided");
  }
}

template <int Dim>
void InitFromFile<Dim>::init(
    int ncid, const zisa::array_view<complex_t, Dim + 1> &u_hat) const {
  const auto varnames = get_varnames(ncid);
  bool contains_u = false;
  bool contains_v = false;
  bool contains_w = false;
  bool contains_omega = false;
  for (const auto &varname : varnames) {
    if (varname == "u") {
      contains_u = true;
    }
    if (varname == "v") {
      contains_v = true;
    }
    if (varname == "w") {
      contains_w = true;
    }
    if (varname == "omega") {
      contains_omega = true;
    }
  }
  if (contains_u && (Dim <= 1 || contains_v) && (Dim <= 2 || contains_w)) {
    read_u_hat(ncid, u_hat);
  } else if (Dim == 2 && contains_omega) {
    const zisa::int_t N_phys = u_hat.shape(1);
    zisa::shape_t<Dim + 1> shape_omega_hat;
    shape_omega_hat[0] = 1;
    for (int d = 0; d < Dim - 1; ++d) {
      shape_omega_hat[d + 1] = N_phys;
    }
    shape_omega_hat[Dim] = N_phys / 2 + 1;
    zisa::array<complex_t, Dim + 1> omega_hat(shape_omega_hat,
                                              zisa::device_type::cpu);
    read_omega_hat(ncid, omega_hat);
    zisa::shape_t<2> omega_hat_view_shape{N_phys, N_phys / 2 + 1};
    zisa::array_view<complex_t, 2> omega_hat_view(
        omega_hat_view_shape, omega_hat.raw(), omega_hat.device());
    zisa::shape_t<3> u_hat_view_shape{2, N_phys, N_phys / 2 + 1};
    zisa::array_view<complex_t, 3> u_hat_view(
        u_hat_view_shape, u_hat.raw(), u_hat.memory_location());
    inverse_curl(omega_hat_view, u_hat_view);
  } else {
    LOG_ERR("Invlid input file provided");
  }
}

template class InitFromFile<1>;
template class InitFromFile<2>;
template class InitFromFile<3>;

}
