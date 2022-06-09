#ifndef AZEBAN_OPERATIONS_ENERGY_SPECTRUM_HPP_
#define AZEBAN_OPERATIONS_ENERGY_SPECTRUM_HPP_

#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <vector>
#include <zisa/memory/array_view.hpp>
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif

namespace azeban {

std::vector<real_t>
energy_spectrum(const Grid<1> &grid,
                const zisa::array_const_view<complex_t, 2> &u_hat);
std::vector<real_t>
energy_spectrum(const Grid<2> &grid,
                const zisa::array_const_view<complex_t, 3> &u_hat);
std::vector<real_t>
energy_spectrum(const Grid<3> &grid,
                const zisa::array_const_view<complex_t, 4> &u_hat);
#if AZEBAN_HAS_MPI
std::vector<real_t>
energy_spectrum(const Grid<1> &grid,
                const zisa::array_const_view<complex_t, 2> &u_hat,
                MPI_Comm comm);
std::vector<real_t>
energy_spectrum(const Grid<2> &grid,
                const zisa::array_const_view<complex_t, 3> &u_hat,
                MPI_Comm comm);
std::vector<real_t>
energy_spectrum(const Grid<3> &grid,
                const zisa::array_const_view<complex_t, 4> &u_hat,
                MPI_Comm comm);
#endif

}

#endif
