#include <azeban/io/absolute_longitudinal_structure_function_writer_factory.hpp>
#include <azeban/io/longitudinal_structure_function_writer_factory.hpp>
#include <azeban/io/second_order_structure_function_writer_factory.hpp>
#include <azeban/io/structure_function_cube_writer_factory.hpp>
#include <azeban/io/structure_function_writer_factory.hpp>
#include <azeban/io/third_order_structure_function_writer_factory.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<Writer<Dim>>
make_structure_function_writer(const nlohmann::json &config,
                               const Grid<Dim> &grid,
                               zisa::int_t sample_idx_start) {
  if (!config.contains("type")) {
    fmt::print(stderr, "Structure function type not provided\n");
    exit(1);
  }

  const std::string type = config["type"];
  if (type == "Second Order") {
    return make_second_order_structure_function_writer<Dim>(
        config, grid, sample_idx_start);
  } else if (type == "Third Order") {
    return make_third_order_structure_function_writer<Dim>(
        config, grid, sample_idx_start);
  } else if (type == "Cube") {
    return make_structure_function_cube_writer<Dim>(
        config, grid, sample_idx_start);
  } else if (type == "Longitudinal") {
    return make_longitudinal_structure_function_writer<Dim>(
        config, grid, sample_idx_start);
  } else if (type == "Absolute Longitudinal") {
    return make_absolute_longitudinal_structure_function_writer<Dim>(
        config, grid, sample_idx_start);
  } else {
    fmt::print(stderr, "Unknown StructureFunctionWriter type \"{}\"\n", type);
    exit(1);
  }
}

#if AZEBAN_HAS_MPI
template <int Dim>
std::unique_ptr<Writer<Dim>>
make_structure_function_writer(const nlohmann::json &config,
                               const Grid<Dim> &grid,
                               zisa::int_t sample_idx_start,
                               const Communicator *comm) {
  if (!config.contains("type")) {
    fmt::print(stderr, "Structure function type not provided\n");
    exit(1);
  }

  const std::string type = config["type"];
  if (type == "Second Order") {
    return make_second_order_structure_function_writer<Dim>(
        config, grid, sample_idx_start, comm);
  } else if (type == "Third Order") {
    fmt::print(stderr,
               "Third Order Structure Function not supported for distributed "
               "computations\n");
    exit(1);
  } else if (type == "Cube") {
    fmt::print(
        stderr,
        "Structure Function Cube not supported for distributed computations\n");
    exit(1);
  } else if (type == "Longitudinal") {
    fmt::print(stderr,
               "Longitudinal Structure Function not supported for distributed "
               "computations\n");
    exit(1);
  } else {
    fmt::print(stderr, "Unknown StructureFunctionWriter type \"{}\"\n", type);
    exit(1);
  }
}
#endif

template std::unique_ptr<Writer<1>> make_structure_function_writer(
    const nlohmann::json &, const Grid<1> &, zisa::int_t);
template std::unique_ptr<Writer<2>> make_structure_function_writer(
    const nlohmann::json &, const Grid<2> &, zisa::int_t);
template std::unique_ptr<Writer<3>> make_structure_function_writer(
    const nlohmann::json &, const Grid<3> &, zisa::int_t);
#if AZEBAN_HAS_MPI
template std::unique_ptr<Writer<1>> make_structure_function_writer(
    const nlohmann::json &, const Grid<1> &, zisa::int_t, const Communicator *);
template std::unique_ptr<Writer<2>> make_structure_function_writer(
    const nlohmann::json &, const Grid<2> &, zisa::int_t, const Communicator *);
template std::unique_ptr<Writer<3>> make_structure_function_writer(
    const nlohmann::json &, const Grid<3> &, zisa::int_t, const Communicator *);
#endif

}
