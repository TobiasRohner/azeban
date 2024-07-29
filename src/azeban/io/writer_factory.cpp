#include <azeban/io/energy_spectrum_writer_factory.hpp>
#include <azeban/io/enstrophy_spectrum_writer_factory.hpp>
#include <azeban/io/netcdf_collective_snapshot_writer_factory.hpp>
#include <azeban/io/netcdf_file_factory.hpp>
#include <azeban/io/netcdf_snapshot_writer_factory.hpp>
#include <azeban/io/structure_function_writer_factory.hpp>
#include <azeban/io/writer_collection.hpp>
#include <azeban/io/writer_factory.hpp>
#if AZEBAN_HAS_CATALYST
#include <azeban/io/paraview_catalyst_writer_factory.hpp>
#endif

namespace azeban {

template <int Dim>
std::unique_ptr<Writer<Dim>> make_writer(const nlohmann::json &config,
                                         const Grid<Dim> &grid,
                                         bool has_tracer,
                                         zisa::int_t num_samples,
                                         zisa::int_t sample_idx_start,
                                         const std::string &full_config,
                                         const std::string &init_script,
                                         void *work_area) {
  if (config.is_array()) {
    auto writer_coll = std::make_unique<WriterCollection<Dim>>(grid);
    for (const auto &writer_config : config) {
      writer_coll->add_writer(make_writer<Dim>(writer_config,
                                               grid,
                                               has_tracer,
                                               num_samples,
                                               sample_idx_start,
                                               full_config,
                                               init_script,
                                               work_area));
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
    } else if (name == "NetCDF Collective") {
      return make_netcdf_collective_snapshot_writer<Dim>(
          config, grid, has_tracer, num_samples, sample_idx_start, work_area);
    }
#if AZEBAN_HAS_CATALYST
    else if (name == "Catalyst") {
      return make_paraview_catalyst_writer<Dim>(config, grid, sample_idx_start);
    }
#endif
    else if (name == "Energy Spectrum") {
      return make_energy_spectrum_writer<Dim>(config, grid, sample_idx_start);
    } else if (name == "Enstrophy Spectrum") {
      return make_enstrophy_spectrum_writer<Dim>(
          config, grid, sample_idx_start);
    } else if (name == "Structure Function") {
      return make_structure_function_writer<Dim>(
          config, grid, sample_idx_start);
    } else if (name == "NetCDF File") {
      // TODO: Add init script as well
      return make_netcdf_file(config,
                              grid,
                              num_samples,
                              sample_idx_start,
                              full_config,
                              init_script);
    } else {
      fmt::print(stderr, "Unknown writer type: \"{}\"\n", name);
      exit(1);
    }
  }
}

#if AZEBAN_HAS_MPI
template <int Dim>
std::unique_ptr<Writer<Dim>> make_writer(const nlohmann::json &config,
                                         const Grid<Dim> &grid,
                                         bool has_tracer,
                                         zisa::int_t num_samples,
                                         zisa::int_t sample_idx_start,
                                         const std::string &full_config,
                                         const std::string &init_script,
                                         const Communicator *comm,
                                         void *work_area) {
  if (config.is_array()) {
    auto writer_coll = std::make_unique<WriterCollection<Dim>>(grid);
    for (const auto &writer_config : config) {
      writer_coll->add_writer(make_writer<Dim>(writer_config,
                                               grid,
                                               has_tracer,
                                               num_samples,
                                               sample_idx_start,
                                               full_config,
                                               init_script,
                                               comm,
                                               work_area));
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
    } else if (name == "NetCDF Collective") {
      return make_netcdf_collective_snapshot_writer<Dim>(
          config, grid, has_tracer, num_samples, sample_idx_start, work_area);
    }
#if AZEBAN_HAS_CATALYST
    else if (name == "Catalyst") {
      return make_paraview_catalyst_writer<Dim>(config, grid, sample_idx_start);
    }
#endif
    else if (name == "Energy Spectrum") {
      return make_energy_spectrum_writer<Dim>(
          config, grid, sample_idx_start, comm);
    } else if (name == "Enstrophy Spectrum") {
      return make_enstrophy_spectrum_writer<Dim>(
          config, grid, sample_idx_start, comm);
    } else if (name == "Second Order Structure Function") {
      return make_structure_function_writer<Dim>(
          config, grid, sample_idx_start, comm);
    } else if (name == "NetCDF File") {
      // TODO: Add init script as well
      return make_netcdf_file(config,
                              grid,
                              num_samples,
                              sample_idx_start,
                              full_config,
                              init_script);
    } else {
      fmt::print(stderr, "Unknown writer type: \"{}\"\n", name);
      exit(1);
    }
  }
}
#endif

template std::unique_ptr<Writer<1>> make_writer(const nlohmann::json &config,
                                                const Grid<1> &grid,
                                                bool has_tracer,
                                                zisa::int_t num_samples,
                                                zisa::int_t sample_idx_start,
                                                const std::string &init_script,
                                                const std::string &,
                                                void *work_area);
template std::unique_ptr<Writer<2>> make_writer(const nlohmann::json &config,
                                                const Grid<2> &grid,
                                                bool has_tracer,
                                                zisa::int_t num_samples,
                                                zisa::int_t sample_idx_start,
                                                const std::string &,
                                                const std::string &init_script,
                                                void *work_area);
template std::unique_ptr<Writer<3>> make_writer(const nlohmann::json &config,
                                                const Grid<3> &grid,
                                                bool has_tracer,
                                                zisa::int_t num_samples,
                                                zisa::int_t sample_idx_start,
                                                const std::string &,
                                                const std::string &init_script,
                                                void *work_area);
#if AZEBAN_HAS_MPI
template std::unique_ptr<Writer<1>> make_writer(const nlohmann::json &config,
                                                const Grid<1> &grid,
                                                bool has_tracer,
                                                zisa::int_t num_samples,
                                                zisa::int_t sample_idx_start,
                                                const std::string &,
                                                const std::string &init_script,
                                                const Communicator *comm,
                                                void *work_area);
template std::unique_ptr<Writer<2>> make_writer(const nlohmann::json &config,
                                                const Grid<2> &grid,
                                                bool has_tracer,
                                                zisa::int_t num_samples,
                                                zisa::int_t sample_idx_start,
                                                const std::string &,
                                                const std::string &init_script,
                                                const Communicator *comm,
                                                void *work_area);
template std::unique_ptr<Writer<3>> make_writer(const nlohmann::json &config,
                                                const Grid<3> &grid,
                                                bool has_tracer,
                                                zisa::int_t num_samples,
                                                zisa::int_t sample_idx_start,
                                                const std::string &,
                                                const std::string &init_script,
                                                const Communicator *comm,
                                                void *work_area);
#endif

}
