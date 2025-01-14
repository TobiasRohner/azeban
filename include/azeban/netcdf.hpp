#ifndef AZEBAN_NETCDF_HPP_
#define AZEBAN_NETCDF_HPP_

#include <azeban/config.hpp>
#include <fmt/core.h>
#include <netcdf.h>
#include <netcdf_par.h>

#define CHECK_NETCDF(...)                                                      \
  if (int status = (__VA_ARGS__); status != NC_NOERR) {                        \
    fmt::print("NetCDF Error {}: {}\n", status, nc_strerror(status));                                   \
    exit(status);                                                              \
  }

namespace azeban {

static constexpr nc_type netcdf_type(float) { return NC_FLOAT; }
static constexpr nc_type netcdf_type(double) { return NC_DOUBLE; }
static constexpr nc_type NC_REAL = netcdf_type(real_t{});

}

#endif
