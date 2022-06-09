#include <azeban/io/energy_spectrum_writer_factory.hpp>
#include <azeban/io/netcdf_snapshot_writer_factory.hpp>
#include <azeban/io/paraview_catalyst_writer_factory.hpp>
#include <azeban/io/writer_collection.hpp>
#include <azeban/io/writer_factory.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<Writer<Dim>> make_writer(const nlohmann::json &config,
                                         const Grid<Dim> &grid,
                                         zisa::int_t sample_idx_start,
                                         void *work_area) {
  if (config.is_array()) {
    auto writer_coll = std::make_unique<WriterCollection<Dim>>(grid);
    for (const auto &writer_config : config) {
      writer_coll->add_writer(
          make_writer<Dim>(writer_config, grid, sample_idx_start, work_area));
    }
    return writer_coll;
  } else {
    if (!config.contains("name")) {
      fmt::print(stderr, "Writer config needs key \"name\"\n");
      exit(1);
    }
    const std::string name = config["name"];
    if (name == "NetCDF Snapshot") {
      return make_netcdf_snapshot_writer<Dim>(
          config, grid, sample_idx_start, work_area);
    } else if (name == "Catalyst") {
      return make_paraview_catalyst_writer<Dim>(config, grid, sample_idx_start);
    } else if (name == "Energy Spectrum") {
      return make_energy_spectrum_writer<Dim>(config, grid, sample_idx_start);
    } else {
      fmt::print(stderr, "Unknown writer type: \"{}\"\n", name);
      exit(1);
    }
  }
}

template std::unique_ptr<Writer<1>> make_writer(const nlohmann::json &config,
                                                const Grid<1> &grid,
                                                zisa::int_t sample_idx_start,
                                                void *work_area);
template std::unique_ptr<Writer<2>> make_writer(const nlohmann::json &config,
                                                const Grid<2> &grid,
                                                zisa::int_t sample_idx_start,
                                                void *work_area);
template std::unique_ptr<Writer<3>> make_writer(const nlohmann::json &config,
                                                const Grid<3> &grid,
                                                zisa::int_t sample_idx_start,
                                                void *work_area);

}
