#ifndef AZEBAN_OPERATIONS_STRUCTURE_FUNCTION_HPP_
#define AZEBAN_OPERATIONS_STRUCTURE_FUNCTION_HPP_

#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <vector>
#include <zisa/memory/array_view.hpp>
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif

namespace azeban {

template<int Dim>
std::vector<real_t>
structure_function_exact(const Grid<Dim> &grid,
                const zisa::array_const_view<complex_t, Dim + 1> &u_hat);
template<int Dim>
std::vector<real_t>
structure_function_approx(const Grid<Dim> &grid,
                const zisa::array_const_view<complex_t, Dim + 1> &u_hat);
#if AZEBAN_HAS_MPI
template<int Dim>
std::vector<real_t>
structure_function_exact(const Grid<Dim> &grid,
                const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                MPI_Comm comm);
template<int Dim>
std::vector<real_t>
structure_function_approx(const Grid<Dim> &grid,
                const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                MPI_Comm comm);
#endif

}

#endif
