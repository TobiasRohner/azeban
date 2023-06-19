#include <azeban/io/third_order_structure_function_writer.hpp>
#include <azeban/io/third_order_structure_function_writer_factory.hpp>
#include <azeban/sequence_factory.hpp>

namespace azeban {

template <int Dim>
static std::unique_ptr<Writer<Dim>>
make_third_order_structure_function_writer(const nlohmann::json &config,
                                           const Grid<Dim> &grid,
                                           zisa::int_t sample_idx_start) {
  if (!config.contains("path")) {
    fmt::print(stderr,
               "ThirdOrderStructureFunctionWriter config needs key \"path\"\n");
    exit(1);
  }
  if (!config.contains("snapshots")) {
    fmt::print(
        stderr,
        "ThirdOrderStructureFunctionWriter config needs key \"snapshots\"\n");
    exit(1);
  }

  const std::string path = config["path"];
  const std::vector<real_t> snapshots
      = make_sequence<real_t>(config["snapshots"]);
  ssize_t max_h = grid.N_phys / 2;
  if (config.contains("maxH")) {
    max_h = config["maxH"];
  }

  return std::make_unique<ThirdOrderStructureFunctionWriter<Dim>>(
      path, grid, snapshots, sample_idx_start, max_h);
}

template std::unique_ptr<Writer<1>> make_third_order_structure_function_writer(
    const nlohmann::json &, const Grid<1> &, zisa::int_t);
template std::unique_ptr<Writer<2>> make_third_order_structure_function_writer(
    const nlohmann::json &, const Grid<2> &, zisa::int_t);
template std::unique_ptr<Writer<3>> make_third_order_structure_function_writer(
    const nlohmann::json &, const Grid<3> &, zisa::int_t);

}
