#ifndef AZEBAN_RUN_FROM_CONFIG_HPP_
#define AZEBAN_RUN_FROM_CONFIG_HPP_

#include <nlohmann/json.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/communicator.hpp>
#endif

namespace azeban {

void run_from_config(const nlohmann::json &config);
#if AZEBAN_HAS_MPI
void run_from_config(const nlohmann::json &config, const Communicator *comm);
#endif

}

#endif
