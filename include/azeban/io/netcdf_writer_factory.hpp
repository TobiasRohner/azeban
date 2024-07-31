#ifndef AZEBAN_IO_NETCDF_WRITER_FACTORY_HPP_
#define AZEBAN_IO_NETCDF_WRITER_FACTORY_HPP_

#include <azeban/io/netcdf_energy_spectrum_writer_factory.hpp>
#include <azeban/io/netcdf_sample_writer_factory.hpp>
#include <azeban/io/netcdf_writer.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<NetCDFWriter<Dim>>
make_netcdf_writer(int ncid,
                   const nlohmann::json &config,
                   const Grid<Dim> &grid,
                   bool has_tracer,
                   zisa::int_t sample_idx_start) {
  if (!config.contains("name")) {
    LOG_ERR("NetCDFWriter config needs key \"name\"");
  }
  const std::string name = config["name"];
  if (name == "Sample") {
    return make_netcdf_sample_writer(
        ncid, config, grid, has_tracer, sample_idx_start);
  } else if (name == "Energy Spectrum") {
    return make_netcdf_energy_spectrum_writer(
        ncid, config, grid, sample_idx_start);
  } else {
    LOG_ERR("Unknown NetCDF Writer type");
  }
}

}

#endif
