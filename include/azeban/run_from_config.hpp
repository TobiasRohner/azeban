#ifndef AZEBAN_RUN_FROM_CONFIG_HPP_
#define AZEBAN_RUN_FROM_CONFIG_HPP_

#include <nlohmann/json.hpp>
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif

namespace azeban {

void run_from_config(const nlohmann::json &config);
#if AZEBAN_HAS_MPI
void run_from_config(const nlohmann::json &config, MPI_Comm comm);
#endif

}

#endif
