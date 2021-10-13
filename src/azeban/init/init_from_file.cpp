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
#include <zisa/io/netcdf_serial_writer.hpp>

namespace azeban {

template <int Dim>
void InitFromFile<Dim>::initialize(const zisa::array_view<real_t, Dim + 1> &u) {
  auto init = [&](auto &&u_) {
    int status, ncid;
    status = nc_open(filename().c_str(), 0, &ncid);
    AZEBAN_ERR_IF(status != NC_NOERR, "Failed to open initial conditions");
    zisa::shape_t<Dim> slice_shape;
    for (int i = 0; i < Dim; ++i) {
      slice_shape[i] = u_.shape(i + 1);
    }
    read_component(ncid,
                   "u",
                   zisa::array_view<real_t, Dim>(
                       slice_shape, u_.raw(), zisa::device_type::cpu));
    if (Dim > 1) {
      read_component(
          ncid,
          "v",
          zisa::array_view<real_t, Dim>(slice_shape,
                                        u_.raw() + zisa::product(slice_shape),
                                        zisa::device_type::cpu));
    }
    if (Dim > 2) {
      read_component(ncid,
                     "w",
                     zisa::array_view<real_t, Dim>(
                         slice_shape,
                         u_.raw() + 2 * zisa::product(slice_shape),
                         zisa::device_type::cpu));
    }
    status = nc_close(ncid);
    AZEBAN_ERR_IF(status != NC_NOERR, "Failed to close file");
  };
  if (u.memory_location() == zisa::device_type::cpu) {
    init(u);
  } else if (u.memory_location() == zisa::device_type::cuda) {
    auto h_u = zisa::array<real_t, Dim + 1>(u.shape(), zisa::device_type::cpu);
    init(h_u);
    zisa::copy(u, h_u);
  } else {
    AZEBAN_ERR("Unknown memory location");
  }
  ++sample_;
}

template <int Dim>
void InitFromFile<Dim>::initialize(
    const zisa::array_view<complex_t, Dim + 1> &u_hat) {
  auto init = [&](auto &&u_) {
    int status, ncid;
    status = nc_open(filename().c_str(), 0, &ncid);
    AZEBAN_ERR_IF(status != NC_NOERR, "Failed to open initial conditions");
    zisa::shape_t<Dim> slice_shape;
    for (int i = 0; i < Dim; ++i) {
      slice_shape[i] = u_.shape(i + 1);
    }
    read_component(ncid,
                   "u",
                   zisa::array_view<real_t, Dim>(
                       slice_shape, u_.raw(), zisa::device_type::cpu));
    if (Dim > 1) {
      read_component(
          ncid,
          "v",
          zisa::array_view<real_t, Dim>(slice_shape,
                                        u_.raw() + zisa::product(slice_shape),
                                        zisa::device_type::cpu));
    }
    if (Dim > 2) {
      read_component(ncid,
                     "w",
                     zisa::array_view<real_t, Dim>(
                         slice_shape,
                         u_.raw() + 2 * zisa::product(slice_shape),
                         zisa::device_type::cpu));
    }
    status = nc_close(ncid);
    AZEBAN_ERR_IF(status != NC_NOERR, "Failed to close file");
  };
  int status, ncid, varid;
  status = nc_open(filename().c_str(), 0, &ncid);
  AZEBAN_ERR_IF(status != NC_NOERR, "Failed to open initial conditions");
  status = nc_inq_varid(ncid, "u", &varid);
  AZEBAN_ERR_IF(status != NC_NOERR, "File does not contain variable \"u\"");
  const auto dims = get_dims(ncid, varid);
  status = nc_close(ncid);
  AZEBAN_ERR_IF(status != NC_NOERR, "Failed to close file");
  zisa::shape_t<Dim + 1> u_shape;
  u_shape[0] = Dim;
  for (int i = 0; i < Dim; ++i) {
    u_shape[i + 1] = dims[i];
  }
  auto u = zisa::array<real_t, Dim + 1>(u_shape, u_hat.memory_location());
  auto fft = make_fft<Dim>(u_hat, u);
  if (u.device() == zisa::device_type::cpu) {
    init(u);
  } else if (u.device() == zisa::device_type::cuda) {
    auto h_u = zisa::array<real_t, Dim + 1>(u.shape(), zisa::device_type::cpu);
    init(h_u);
    zisa::copy(u, h_u);
  } else {
    AZEBAN_ERR("Unknown memory location");
  }
  fft->forward();
  ++sample_;
}

template <int Dim>
std::string InitFromFile<Dim>::filename() const {
  return experiment_ + "/sample_" + std::to_string(sample_) + "_time_" + time_
         + ".nc";
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

template class InitFromFile<1>;
template class InitFromFile<2>;
template class InitFromFile<3>;

}
