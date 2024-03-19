#ifndef AZEBAN_IO_WRITER_FACTORY_HPP_
#define AZEBAN_IO_WRITER_FACTORY_HPP_

#include <azeban/grid.hpp>
#include <azeban/io/writer.hpp>
#include <nlohmann/json.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/communicator.hpp>
#endif

namespace azeban {

template <int Dim>
std::unique_ptr<Writer<Dim>> make_writer(const nlohmann::json &config,
                                         const Grid<Dim> &grid,
                                         bool has_tracer,
                                         zisa::int_t num_samples,
                                         zisa::int_t sample_idx_start,
                                         void *work_area = nullptr);
#if AZEBAN_HAS_MPI
template <int Dim>
std::unique_ptr<Writer<Dim>> make_writer(const nlohmann::json &config,
                                         const Grid<Dim> &grid,
                                         bool has_tracer,
                                         zisa::int_t num_samples,
                                         zisa::int_t sample_idx_start,
                                         const Communicator *comm,
                                         void *work_area = nullptr);
#endif

}

#endif
