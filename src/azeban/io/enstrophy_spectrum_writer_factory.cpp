#include <azeban/io/enstrophy_spectrum_writer_factory.hpp>
#include <azeban/sequence_factory.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<EnstrophySpectrumWriter<Dim>>
make_enstrophy_spectrum_writer(const nlohmann::json &config,
                               const Grid<Dim> &grid,
                               zisa::int_t sample_idx_start) {
  if (!config.contains("path")) {
    fmt::print(stderr, "EnstrophySpectrumWriter config needs key \"path\"\n");
    exit(1);
  }
  if (!config.contains("snapshots")) {
    fmt::print(stderr,
               "EnstrophySpectrumWriter config needs key \"snapshots\"\n");
    exit(1);
  }

  const std::string path = config["path"];
  const std::vector<double> snapshots
      = make_sequence<double>(config["snapshots"]);

  return std::make_unique<EnstrophySpectrumWriter<Dim>>(
      path, grid, snapshots, sample_idx_start);
}

#if AZEBAN_HAS_MPI
template <int Dim>
std::unique_ptr<EnstrophySpectrumWriter<Dim>>
make_enstrophy_spectrum_writer(const nlohmann::json &config,
                               const Grid<Dim> &grid,
                               zisa::int_t sample_idx_start,
                               const Communicator *comm) {
  if (!config.contains("path")) {
    fmt::print(stderr, "EnstrophySpectrumWriter config needs key \"path\"\n");
    exit(1);
  }
  if (!config.contains("snapshots")) {
    fmt::print(stderr,
               "EnstrophySpectrumWriter config needs key \"snapshots\"\n");
    exit(1);
  }

  const std::string path = config["path"];
  const std::vector<double> snapshots
      = make_sequence<double>(config["snapshots"]);

  return std::make_unique<EnstrophySpectrumWriter<Dim>>(
      path, grid, snapshots, sample_idx_start, comm);
}
#endif

template std::unique_ptr<EnstrophySpectrumWriter<1>>
make_enstrophy_spectrum_writer<1>(const nlohmann::json &,
                                  const Grid<1> &,
                                  zisa::int_t);
template std::unique_ptr<EnstrophySpectrumWriter<2>>
make_enstrophy_spectrum_writer<2>(const nlohmann::json &,
                                  const Grid<2> &,
                                  zisa::int_t);
template std::unique_ptr<EnstrophySpectrumWriter<3>>
make_enstrophy_spectrum_writer<3>(const nlohmann::json &,
                                  const Grid<3> &,
                                  zisa::int_t);
#if AZEBAN_HAS_MPI
template std::unique_ptr<EnstrophySpectrumWriter<1>>
make_enstrophy_spectrum_writer<1>(const nlohmann::json &,
                                  const Grid<1> &,
                                  zisa::int_t,
                                  const Communicator *);
template std::unique_ptr<EnstrophySpectrumWriter<2>>
make_enstrophy_spectrum_writer<2>(const nlohmann::json &,
                                  const Grid<2> &,
                                  zisa::int_t,
                                  const Communicator *);
template std::unique_ptr<EnstrophySpectrumWriter<3>>
make_enstrophy_spectrum_writer<3>(const nlohmann::json &,
                                  const Grid<3> &,
                                  zisa::int_t,
                                  const Communicator *);
#endif

}
