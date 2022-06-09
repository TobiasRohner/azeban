#include <azeban/io/energy_spectrum_writer_factory.hpp>
#include <azeban/sequence_factory.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<EnergySpectrumWriter<Dim>>
make_energy_spectrum_writer(const nlohmann::json &config,
                            const Grid<Dim> &grid,
                            zisa::int_t sample_idx_start) {
  if (!config.contains("path")) {
    fmt::print(stderr, "EnergySpectrumWriter config needs key \"path\"\n");
    exit(1);
  }
  if (!config.contains("snapshots")) {
    fmt::print(stderr, "EnergySpectrumWriter config needs key \"snapshots\"\n");
    exit(1);
  }

  const std::string path = config["path"];
  const std::vector<real_t> snapshots
      = make_sequence<real_t>(config["snapshots"]);

  return std::make_unique<EnergySpectrumWriter<Dim>>(
      path, grid, snapshots, sample_idx_start);
}

template std::unique_ptr<EnergySpectrumWriter<1>>
make_energy_spectrum_writer<1>(const nlohmann::json &,
                               const Grid<1> &,
                               zisa::int_t);
template std::unique_ptr<EnergySpectrumWriter<2>>
make_energy_spectrum_writer<2>(const nlohmann::json &,
                               const Grid<2> &,
                               zisa::int_t);
template std::unique_ptr<EnergySpectrumWriter<3>>
make_energy_spectrum_writer<3>(const nlohmann::json &,
                               const Grid<3> &,
                               zisa::int_t);

}
