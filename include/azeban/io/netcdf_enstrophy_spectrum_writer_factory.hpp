#ifndef AZEBAN_IO_NETCDF_ENSTROPHY_SPECTRUM_WRITER_FACTORY_HPP_
#define AZEBAN_IO_NETCDF_ENSTROPHY_SPECTRUM_WRITER_FACTORY_HPP_

#include <azeban/io/netcdf_enstrophy_spectrum_writer.hpp>
#include <azeban/sequence_factory.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<NetCDFEnstrophySpectrumWriter<Dim>>
make_netcdf_enstrophy_spectrum_writer(int ncid,
                                      const nlohmann::json &config,
                                      const Grid<Dim> &grid,
                                      zisa::int_t sample_idx_start) {
  if (!config.contains("snapshots")) {
    LOG_ERR("NetCDFEnstrophySpectrumWriter config needs key \"snapshots\"");
  }
  const std::vector<double> snapshots
      = make_sequence<double>(config["snapshots"]);
  return std::make_unique<NetCDFEnstrophySpectrumWriter<Dim>>(
      ncid, grid, snapshots, sample_idx_start);
}

}

#endif
