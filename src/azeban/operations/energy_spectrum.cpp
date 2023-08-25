#include <azeban/operations/energy_spectrum.hpp>
#include <azeban/operations/spectrum.hpp>
#include <iostream>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/mpi_types.hpp>
#endif

namespace azeban {

struct EnergySpectrumOperator {
  static real_t eval(const Grid<1> &grid, long, complex_t u) {
    const real_t norm = 1. / zisa::pow<1>(grid.N_phys);
    return 0.5 * abs2(norm * u);
  }

  static real_t
  eval(const Grid<2> &grid, long, long, complex_t u, complex_t v) {
    const real_t norm = 1. / zisa::pow<2>(grid.N_phys);
    return 0.5 * (abs2(norm * u) + abs2(norm * v));
  }

  static real_t eval(const Grid<3> &grid,
                     long,
                     long,
                     long,
                     complex_t u,
                     complex_t v,
                     complex_t w) {
    const real_t norm = 1. / zisa::pow<3>(grid.N_phys);
    return 0.5 * (abs2(norm * u) + abs2(norm * v) + abs2(norm * w));
  }
};

std::vector<real_t>
energy_spectrum(const Grid<1> &grid,
                const zisa::array_const_view<complex_t, 2> &u_hat) {
  return spectrum<EnergySpectrumOperator>(grid, u_hat);
}

std::vector<real_t>
energy_spectrum(const Grid<2> &grid,
                const zisa::array_const_view<complex_t, 3> &u_hat) {
  return spectrum<EnergySpectrumOperator>(grid, u_hat);
}

std::vector<real_t>
energy_spectrum(const Grid<3> &grid,
                const zisa::array_const_view<complex_t, 4> &u_hat) {
  return spectrum<EnergySpectrumOperator>(grid, u_hat);
}

#if AZEBAN_HAS_MPI
std::vector<real_t>
energy_spectrum(const Grid<1> &grid,
                const zisa::array_const_view<complex_t, 2> &u_hat,
                MPI_Comm comm) {
  return spectrum<EnergySpectrumOperator>(grid, u_hat, comm);
}

std::vector<real_t>
energy_spectrum(const Grid<2> &grid,
                const zisa::array_const_view<complex_t, 3> &u_hat,
                MPI_Comm comm) {
  return spectrum<EnergySpectrumOperator>(grid, u_hat, comm);
}

std::vector<real_t>
energy_spectrum(const Grid<3> &grid,
                const zisa::array_const_view<complex_t, 4> &u_hat,
                MPI_Comm comm) {
  return spectrum<EnergySpectrumOperator>(grid, u_hat, comm);
}
#endif

}
