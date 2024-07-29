#include <azeban/io/netcdf_file_factory.hpp>
#include <azeban/io/netcdf_writer_factory.hpp>
#include <azeban/sequence_factory.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<NetCDFFile<Dim>>
make_netcdf_file(const nlohmann::json &config,
                 const Grid<Dim> &grid,
                 zisa::int_t num_samples,
                 zisa::int_t sample_idx_start,
                 const std::string &full_config,
                 const std::string &init_script) {
  if (!config.contains("path")) {
    fmt::print(stderr, "NetCDFFile config needs key \"path\"\n");
    exit(1);
  }
  const std::string path = config["path"];
  auto file = std::make_unique<NetCDFFile<Dim>>(
      path, grid, num_samples, sample_idx_start, full_config, init_script);
  if (config.contains("contents")) {
    for (const auto &cont : config["contents"]) {
      file->add_writer(
          make_netcdf_writer(file->ncid(), cont, grid, sample_idx_start));
    }
  }
  return file;
}

template std::unique_ptr<NetCDFFile<1>>
make_netcdf_file<1>(const nlohmann::json &config,
                    const Grid<1> &grid,
                    zisa::int_t num_samples,
                    zisa::int_t sample_idx_start,
                    const std::string &full_config,
                    const std::string &init_script);
template std::unique_ptr<NetCDFFile<2>>
make_netcdf_file<2>(const nlohmann::json &config,
                    const Grid<2> &grid,
                    zisa::int_t num_samples,
                    zisa::int_t sample_idx_start,
                    const std::string &full_config,
                    const std::string &init_script);
template std::unique_ptr<NetCDFFile<3>>
make_netcdf_file<3>(const nlohmann::json &config,
                    const Grid<3> &grid,
                    zisa::int_t num_samples,
                    zisa::int_t sample_idx_start,
                    const std::string &full_config,
                    const std::string &init_script);

}
