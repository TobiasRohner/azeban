#ifndef AZEBAN_IO_NETCDF_COLLECTIVE_SNAPSHOT_WRITER_FACTORY_HPP_
#define AZEBAN_IO_NETCDF_COLLECTIVE_SNAPSHOT_WRITER_FACTORY_HPP_

#include <azeban/io/netcdf_collective_snapshot_writer.hpp>
#include <azeban/mpi/communicator.hpp>
#include <memory>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<NetCDFCollectiveSnapshotWriter<Dim>>
make_netcdf_collective_snapshot_writer(const nlohmann::json &config,
                                       const Grid<Dim> &grid,
                                       bool has_tracer,
                                       zisa::int_t num_samples,
                                       zisa::int_t sample_idx_start,
                                       void *wrk_area = nullptr);

}

#endif
