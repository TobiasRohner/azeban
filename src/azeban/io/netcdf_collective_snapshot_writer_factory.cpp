#include <azeban/io/netcdf_collective_snapshot_writer_factory.hpp>
#include <azeban/sequence_factory.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<NetCDFCollectiveSnapshotWriter<Dim>>
make_netcdf_collective_snapshot_writer(const nlohmann::json &config,
                                       const Grid<Dim> &grid,
                                       bool has_tracer,
                                       zisa::int_t num_samples,
                                       zisa::int_t sample_idx_start,
                                       void *work_area) {
  if (!config.contains("path")) {
    fmt::print(stderr,
               "NetCDFCollectiveSnapshotWriter config needs key \"path\"\n");
    exit(1);
  }
  if (!config.contains("snapshots")) {
    fmt::print(
        stderr,
        "NetCDFCollectiveSnapshotWriter config needs key \"snapshots\"\n");
    exit(1);
  }

  const std::string path = config["path"];
  const std::vector<real_t> snapshots
      = make_sequence<real_t>(config["snapshots"]);
  bool save_pressure = false;
  if (config.contains("save_pressure")) {
    save_pressure = config["save_pressure"];
  }

  return std::make_unique<NetCDFCollectiveSnapshotWriter<Dim>>(path,
                                                               grid,
                                                               snapshots,
                                                               has_tracer,
                                                               num_samples,
                                                               sample_idx_start,
                                                               save_pressure,
                                                               work_area);
}

template std::unique_ptr<NetCDFCollectiveSnapshotWriter<1>>
make_netcdf_collective_snapshot_writer<1>(const nlohmann::json &config,
                                          const Grid<1> &grid,
                                          bool has_tracer,
                                          zisa::int_t num_samples,
                                          zisa::int_t sample_idx_start,
                                          void *wok_area);
template std::unique_ptr<NetCDFCollectiveSnapshotWriter<2>>
make_netcdf_collective_snapshot_writer<2>(const nlohmann::json &config,
                                          const Grid<2> &grid,
                                          bool has_tracer,
                                          zisa::int_t num_samples,
                                          zisa::int_t sample_idx_start,
                                          void *wok_area);
template std::unique_ptr<NetCDFCollectiveSnapshotWriter<3>>
make_netcdf_collective_snapshot_writer<3>(const nlohmann::json &config,
                                          const Grid<3> &grid,
                                          bool has_tracer,
                                          zisa::int_t num_samples,
                                          zisa::int_t sample_idx_start,
                                          void *wok_area);

}
