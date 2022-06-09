#ifndef AZEBAN_IO_ENERGY_SPECTRUM_WRITER_FACTORY_HPP_
#define AZEBAN_IO_ENERGY_SPECTRUM_WRITER_FACTORY_HPP_

#include <azeban/io/energy_spectrum_writer.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/communicator.hpp>
#endif

namespace azeban {

template <int Dim>
std::unique_ptr<EnergySpectrumWriter<Dim>>
make_energy_spectrum_writer(const nlohmann::json &config,
                            const Grid<Dim> &grid,
                            zisa::int_t sample_idx_start);
#if AZEBAN_HAS_MPI
template <int Dim>
std::unique_ptr<EnergySpectrumWriter<Dim>>
make_energy_spectrum_writer(const nlohmann::json &config,
                            const Grid<Dim> &grid,
                            zisa::int_t sample_idx_start,
                            const Communicator *comm);
#endif

}

#endif
