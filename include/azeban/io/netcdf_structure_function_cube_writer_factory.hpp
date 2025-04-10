#ifndef AZEBAN_IO_NETCDF_STRUCTURE_FUNCTION_CUBE_WRITER_FACTORY_HPP_
#define AZEBAN_IO_NETCDF_STRUCTURE_FUNCTION_CUBE_WRITER_FACTORY_HPP_

#include <azeban/io/netcdf_structure_function_cube_writer.hpp>
#include <azeban/operations/structure_function_functionals.hpp>
#include <azeban/sequence_factory.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<NetCDFWriter<Dim>>
make_netcdf_structure_function_cube_sf_cube_writer(
    int ncid,
    const nlohmann::json &config,
    const Grid<Dim> &grid,
    const std::vector<double> &snapshots,
    zisa::int_t sample_idx_start) {
  if (!config.contains("p")) {
    LOG_ERR(
        "NetCDFStructureFunctionCubeWriter config for SFCube needs key \"p\"");
  }
  const real_t p = config["p"];
  zisa::int_t max_h = (grid.N_phys + 1) / 2;
  if (config.contains("max_h")) {
    max_h = config["max_h"];
  }
  return std::make_unique<
      NetCDFStructureFunctionCubeWriter<Dim, SFCubeFunctional>>(
      ncid,
      grid,
      snapshots,
      "structure_function_cube_" + std::to_string(p),
      SFCubeFunctional(p),
      max_h,
      sample_idx_start);
}

template <int Dim>
std::unique_ptr<NetCDFWriter<Dim>>
make_netcdf_structure_function_cube_third_order_writer(
    int ncid,
    const nlohmann::json &config,
    const Grid<Dim> &grid,
    const std::vector<double> &snapshots,
    zisa::int_t sample_idx_start) {
  zisa::int_t max_h = (grid.N_phys + 1) / 2;
  if (config.contains("max_h")) {
    max_h = config["max_h"];
  }
  return std::make_unique<
      NetCDFStructureFunctionCubeWriter<Dim, SFThirdOrderFunctional>>(
      ncid,
      grid,
      snapshots,
      "third_order_structure_function",
      SFThirdOrderFunctional(),
      max_h,
      sample_idx_start);
}

template <int Dim>
std::unique_ptr<NetCDFWriter<Dim>>
make_netcdf_structure_function_cube_longitudinal_writer(
    int ncid,
    const nlohmann::json &config,
    const Grid<Dim> &grid,
    const std::vector<double> &snapshots,
    zisa::int_t sample_idx_start) {
  if (!config.contains("p")) {
    LOG_ERR("NetCDFStructureFunctionCubeWriter config for Longitudinal needs "
            "key \"p\"");
  }
  const real_t p = config["p"];
  zisa::int_t max_h = (grid.N_phys + 1) / 2;
  if (config.contains("max_h")) {
    max_h = config["max_h"];
  }
  return std::make_unique<
      NetCDFStructureFunctionCubeWriter<Dim, SFLongitudinalFunctional>>(
      ncid,
      grid,
      snapshots,
      "structure_function_longitudinal_" + std::to_string(p),
      SFLongitudinalFunctional(p),
      max_h,
      sample_idx_start);
}

template <int Dim>
std::unique_ptr<NetCDFWriter<Dim>>
make_netcdf_structure_function_cube_absolute_longitudinal_writer(
    int ncid,
    const nlohmann::json &config,
    const Grid<Dim> &grid,
    const std::vector<double> &snapshots,
    zisa::int_t sample_idx_start) {
  if (!config.contains("p")) {
    LOG_ERR("NetCDFStructureFunctionCubeWriter config for Absolute "
            "Longitudinal needs key \"p\"");
  }
  const real_t p = config["p"];
  zisa::int_t max_h = (grid.N_phys + 1) / 2;
  if (config.contains("max_h")) {
    max_h = config["max_h"];
  }
  return std::make_unique<
      NetCDFStructureFunctionCubeWriter<Dim, SFAbsoluteLongitudinalFunctional>>(
      ncid,
      grid,
      snapshots,
      "structure_function_absolute_longitudinal_" + std::to_string(p),
      SFAbsoluteLongitudinalFunctional(p),
      max_h,
      sample_idx_start);
}

template <int Dim>
std::unique_ptr<NetCDFWriter<Dim>>
make_netcdf_structure_function_cube_writer(int ncid,
                                           const nlohmann::json &config,
                                           const Grid<Dim> &grid,
                                           zisa::int_t sample_idx_start) {
  if (!config.contains("type")) {
    LOG_ERR("NetCDFStructureFunctionCubeWriter config needs key \"type\"");
  }
  const std::string type = config["type"];
  if (!config.contains("snapshots")) {
    LOG_ERR("NetCDFStructureFunctionCubeWriter config needs key \"snapshots\"");
  }
  const std::vector<double> snapshots
      = make_sequence<double>(config["snapshots"]);
  if (type == "SFCube") {
    return make_netcdf_structure_function_cube_sf_cube_writer(
        ncid, config, grid, snapshots, sample_idx_start);
  } else if (type == "Third Order") {
    return make_netcdf_structure_function_cube_third_order_writer(
        ncid, config, grid, snapshots, sample_idx_start);
  } else if (type == "Longitudinal") {
    return make_netcdf_structure_function_cube_longitudinal_writer(
        ncid, config, grid, snapshots, sample_idx_start);
  } else if (type == "Absolute Longitudinal") {
    return make_netcdf_structure_function_cube_absolute_longitudinal_writer(
        ncid, config, grid, snapshots, sample_idx_start);
  } else {
    LOG_ERR("Unsupported structure function type provided in key \"type\"");
  }
}

}

#endif
