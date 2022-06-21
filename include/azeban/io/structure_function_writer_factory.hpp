#ifndef AZEBAN_IO_STRUCTURE_FUNCTION_WRITER_FACTORY_HPP_
#define AZEBAN_IO_STRUCTURE_FUNCTION_WRITER_FACTORY_HPP_

#include <azeban/io/writer.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/communicator.hpp>
#endif

namespace azeban {

template <int Dim>
std::unique_ptr<Writer<Dim>>
make_structure_function_writer(const nlohmann::json &config,
                               const Grid<Dim> &grid,
                               zisa::int_t sample_idx_start);
#if AZEBAN_HAS_MPI
template <int Dim>
std::unique_ptr<Writer<Dim>>
make_structure_function_writer(const nlohmann::json &config,
                               const Grid<Dim> &grid,
                               zisa::int_t sample_idx_start,
                               const Communicator *comm);
#endif

}

#endif
