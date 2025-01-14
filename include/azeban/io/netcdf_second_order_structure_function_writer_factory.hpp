#ifndef AZEBAN_IO_NETCDF_SECOND_ORDER_STRUCTURE_FUNCTION_WRITER_FACTORY_HPP_
#define AZEBAN_IO_NETCDF_SECOND_ORDER_STRUCTURE_FUNCTION_WRITER_FACTORY_HPP_

#include <azeban/io/netcdf_second_order_structure_function_writer.hpp>
#include <azeban/sequence_factory.hpp>


namespace azeban {

template <int Dim>
std::unique_ptr<NetCDFWriter<Dim>>
make_netcdf_second_order_structure_function_writer(int ncid,
						   const nlohmann::json &config,
						   const Grid<Dim> &grid,
						   zisa::int_t sample_idx_start) {
  if (!config.contains("snapshots")) {
    LOG_ERR("NetCDFSecondOrderStructureFunctionWriter config needs key \"snapshots\"");
  }
  const std::vector<real_t> snapshots = make_sequence<real_t>(config["snapshots"]);
  bool exact = false;
  if (config.contains("exact")) {
    exact = config["exact"];
  }

  if (exact) {
    return std::make_unique<NetCDFSecondOrderStructureFunctionWriter<Dim, detail::SF2Exact>>(ncid, grid, snapshots, sample_idx_start);
  } else {
    return std::make_unique<NetCDFSecondOrderStructureFunctionWriter<Dim, detail::SF2Approx>>(ncid, grid, snapshots, sample_idx_start);
  }
}

}


#endif
