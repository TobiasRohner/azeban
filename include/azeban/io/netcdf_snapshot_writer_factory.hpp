#ifndef AZEBAN_IO_NETCDF_SNAPSHOT_WRITER_FACTORY_HPP_
#define AZEBAN_IO_NETCDF_SNAPSHOT_WRITER_FACTORY_HPP_

#include <azeban/io/netcdf_snapshot_writer.hpp>
#include <memory>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<NetCDFSnapshotWriter<Dim>>
make_netcdf_snapshot_writer(const nlohmann::json &config,
                            const Grid<Dim> &grid,
                            void *wrk_area = nullptr);

}

#endif
