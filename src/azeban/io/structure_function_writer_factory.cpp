#include <azeban/io/structure_function_writer_factory.hpp>
#include <azeban/io/structure_function_writer.hpp>
#include <azeban/sequence_factory.hpp>

namespace azeban {

template <int Dim, typename WRITER>
static std::unique_ptr<WRITER>
make_structure_function_writer_type(const nlohmann::json &config,
                            const Grid<Dim> &grid,
                            zisa::int_t sample_idx_start) {
  if (!config.contains("path")) {
    fmt::print(stderr, "StructureFunctionWriter config needs key \"path\"\n");
    exit(1);
  }
  if (!config.contains("snapshots")) {
    fmt::print(stderr, "StructureFunctionWriter config needs key \"snapshots\"\n");
    exit(1);
  }

  const std::string path = config["path"];
  const std::vector<real_t> snapshots
      = make_sequence<real_t>(config["snapshots"]);

  return std::make_unique<WRITER>(
      path, grid, snapshots, sample_idx_start);
}

template <int Dim>
static std::unique_ptr<Writer<Dim>>
make_structure_function_writer(const nlohmann::json &config,
                            const Grid<Dim> &grid,
                            zisa::int_t sample_idx_start) {
  bool exact = false;
  if (config.contains("exact")) {
    exact = config["exact"];
  }

  if (exact) {
    return make_structure_function_writer_type<Dim, StructureFunctionWriterExact<Dim>>(config, grid, sample_idx_start);
  }
  else {
    return make_structure_function_writer_type<Dim, StructureFunctionWriterApprox<Dim>>(config, grid, sample_idx_start);
  }
}

#if AZEBAN_HAS_MPI
template <int Dim, typename WRITER>
static std::unique_ptr<WRITER>
make_structure_function_writer_type(const nlohmann::json &config,
                            const Grid<Dim> &grid,
                            zisa::int_t sample_idx_start,
                            const Communicator *comm) {
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

  return std::make_unique<WRITER>(
      path, grid, snapshots, sample_idx_start, comm);
}

template <int Dim>
static std::unique_ptr<Writer<Dim>>
make_structure_function_writer(const nlohmann::json &config,
                            const Grid<Dim> &grid,
                            zisa::int_t sample_idx_start,
			    const Communicator *comm) {
  bool exact = false;
  if (config.contains("exact")) {
    exact = config["exact"];
  }

  if (exact) {
    return make_structure_function_writer_type<Dim, StructureFunctionWriterExact<Dim>>(config, grid, sample_idx_start, comm);
  }
  else {
    return make_structure_function_writer_type<Dim, StructureFunctionWriterApprox<Dim>>(config, grid, sample_idx_start, comm);
  }
}
#endif

template std::unique_ptr<Writer<1>>
make_structure_function_writer<1>(const nlohmann::json &,
                               const Grid<1> &,
                               zisa::int_t);
template std::unique_ptr<Writer<2>>
make_structure_function_writer<2>(const nlohmann::json &,
                               const Grid<2> &,
                               zisa::int_t);
template std::unique_ptr<Writer<3>>
make_structure_function_writer<3>(const nlohmann::json &,
                               const Grid<3> &,
                               zisa::int_t);
#if AZEBAN_HAS_MPI
template std::unique_ptr<Writer<1>>
make_structure_function_writer<1>(const nlohmann::json &,
                               const Grid<1> &,
                               zisa::int_t,
                               const Communicator *);
template std::unique_ptr<Writer<2>>
make_structure_function_writer<2>(const nlohmann::json &,
                               const Grid<2> &,
                               zisa::int_t,
                               const Communicator *);
template std::unique_ptr<Writer<3>>
make_structure_function_writer<3>(const nlohmann::json &,
                               const Grid<3> &,
                               zisa::int_t,
                               const Communicator *);
#endif

}
