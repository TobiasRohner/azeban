#ifndef AZEBAN_IO_NETCDF_FILE_FACTORY_HPP_
#define AZEBAN_IO_NETCDF_FILE_FACTORY_HPP_

#include <azeban/io/netcdf_file.hpp>
#include <azeban/mpi/communicator.hpp>
#include <memory>
#include <nlohmann/json.hpp>

namespace azeban {

template <int Dim>
std::unique_ptr<NetCDFFile<Dim>>
make_netcdf_file(const nlohmann::json &writer_config,
                 const Grid<Dim> &grid,
                 bool has_tracer,
                 zisa::int_t num_samples,
                 zisa::int_t sample_idx_start,
                 const std::string &full_config,
                 const std::string &init_script);

#if AZEBAN_HAS_MPI
template <int Dim>
std::unique_ptr<NetCDFFile<Dim>>
make_netcdf_file(const nlohmann::json &writer_config,
                 const Grid<Dim> &grid,
                 bool has_tracer,
                 zisa::int_t num_samples,
                 zisa::int_t sample_idx_start,
                 const std::string &full_config,
                 const std::string &init_script,
                 const Communicator *comm);
#endif

}

#endif
