#include <azeban/io/paraview_catalyst_writer_factory.hpp>
#include <azeban/sequence_factory.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<ParaviewCatalystWriter<Dim>>
make_paraview_catalyst_writer(const nlohmann::json &config,
                              const Grid<Dim> &grid,
                              zisa::int_t sample_idx_start) {
  if (!config.contains("scripts")) {
    fmt::print(stderr, "ParaviewCatalystWriter config needs key \"scripts\"\n");
    exit(1);
  }
  if (!config.contains("snapshots")) {
    fmt::print(stderr,
               "ParaviewCatalystWriter config needs key \"snapshots\"\n");
    exit(1);
  }

  const std::vector<std::string> scripts
      = config["scripts"].get<std::vector<std::string>>();
  const std::vector<real_t> snapshots
      = make_sequence<real_t>(config["snapshots"]);

  return std::make_unique<ParaviewCatalystWriter<Dim>>(
      grid, snapshots, scripts, sample_idx_start);
}

template std::unique_ptr<ParaviewCatalystWriter<1>>
make_paraview_catalyst_writer<1>(const nlohmann::json &,
                                 const Grid<1> &,
                                 zisa::int_t);
template std::unique_ptr<ParaviewCatalystWriter<2>>
make_paraview_catalyst_writer<2>(const nlohmann::json &,
                                 const Grid<2> &,
                                 zisa::int_t);
template std::unique_ptr<ParaviewCatalystWriter<3>>
make_paraview_catalyst_writer<3>(const nlohmann::json &,
                                 const Grid<3> &,
                                 zisa::int_t);

}
