#include <azeban/io/netcdf_snapshot_writer_factory.hpp>
#include <azeban/sequence_factory.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<NetCDFSnapshotWriter<Dim>>
make_netcdf_snapshot_writer(const nlohmann::json &config,
                            const Grid<Dim> &grid,
                            zisa::int_t sample_idx_start,
                            void *work_area) {
  if (!config.contains("path")) {
    fmt::print(stderr, "NetCDFSnapshotWriter config needs key \"path\"\n");
    exit(1);
  }
  if (!config.contains("snapshots")) {
    fmt::print(stderr, "NetCDFSnapshotWriter config needs key \"snapshots\"\n");
    exit(1);
  }

  const std::string path = config["path"];
  const std::vector<real_t> snapshots
      = make_sequence<real_t>(config["snapshots"]);
  bool store_u_hat = false;
  if (config.contains("fourier")) {
    store_u_hat = config["fourier"];
  }

  return std::make_unique<NetCDFSnapshotWriter<Dim>>(
      path, store_u_hat, grid, snapshots, sample_idx_start, work_area);
}

template std::unique_ptr<NetCDFSnapshotWriter<1>>
make_netcdf_snapshot_writer<1>(const nlohmann::json &config,
                               const Grid<1> &grid,
                               zisa::int_t sample_idx_start,
                               void *wok_area);
template std::unique_ptr<NetCDFSnapshotWriter<2>>
make_netcdf_snapshot_writer<2>(const nlohmann::json &config,
                               const Grid<2> &grid,
                               zisa::int_t sample_idx_start,
                               void *wok_area);
template std::unique_ptr<NetCDFSnapshotWriter<3>>
make_netcdf_snapshot_writer<3>(const nlohmann::json &config,
                               const Grid<3> &grid,
                               zisa::int_t sample_idx_start,
                               void *wok_area);

}
