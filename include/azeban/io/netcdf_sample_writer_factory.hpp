#ifndef AZEBAN_IO_NETCDF_SAMPLE_WRITER_FACTORY_HPP_
#define AZEBAN_IO_NETCDF_SAMPLE_WRITER_FACTORY_HPP_

#include <azeban/io/netcdf_sample_writer.hpp>
#include <azeban/sequence_factory.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<NetCDFSampleWriter<Dim>>
make_netcdf_sample_writer(int ncid,
                          const nlohmann::json &config,
                          const Grid<Dim> &grid,
                          zisa::int_t sample_idx_start) {
  if (!config.contains("snapshots")) {
    LOG_ERR("NetCDFSampleWriter config needs key \"snapshots\"");
  }
  const std::vector<real_t> snapshots
      = make_sequence<real_t>(config["snapshots"]);
  zisa::int_t N = grid.N_phys;
  if (config.contains("N")) {
    N = config["N"];
  }
  return std::make_unique<NetCDFSampleWriter<Dim>>(
      ncid, grid, N, snapshots, sample_idx_start);
}

}

#endif
