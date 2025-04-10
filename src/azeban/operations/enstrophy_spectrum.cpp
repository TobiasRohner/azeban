#include <azeban/operations/enstrophy_spectrum.hpp>
#include <azeban/operations/spectrum.hpp>
#include <iostream>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/mpi_types.hpp>
#endif

namespace azeban {

struct EnstrophySpectrumOperator {
  static real_t eval(const Grid<1> &, long, complex_t) { return 0; }

  static real_t
  eval(const Grid<2> &grid, long k1, long k2, complex_t u, complex_t v) {
    const complex_t curl_z
        = 2 * zisa::pi * (complex_t(0, k1) * v - complex_t(0, k2) * u);
    const real_t norm = 1. / zisa::pow<2>(grid.N_phys);
    return 0.5 * abs2(norm * curl_z);
  }

  static real_t eval(const Grid<3> &grid,
                     long k1,
                     long k2,
                     long k3,
                     complex_t u,
                     complex_t v,
                     complex_t w) {
    const complex_t curl_x
        = 2 * zisa::pi * (complex_t(0, k2) * w - complex_t(0, k3) * v);
    const complex_t curl_y
        = 2 * zisa::pi * (complex_t(0, k3) * u - complex_t(0, k1) * w);
    const complex_t curl_z
        = 2 * zisa::pi * (complex_t(0, k1) * v - complex_t(0, k2) * u);
    const real_t norm = 1. / zisa::pow<3>(grid.N_phys);
    return 0.5
           * (abs2(norm * curl_x) + abs2(norm * curl_y) + abs2(norm * curl_z));
  }
};

std::vector<real_t>
enstrophy_spectrum(const Grid<1> &grid,
                   const zisa::array_const_view<complex_t, 2> &u_hat) {
  return spectrum<EnstrophySpectrumOperator>(grid, u_hat);
}

std::vector<real_t>
enstrophy_spectrum(const Grid<2> &grid,
                   const zisa::array_const_view<complex_t, 3> &u_hat) {
  return spectrum<EnstrophySpectrumOperator>(grid, u_hat);
}

std::vector<real_t>
enstrophy_spectrum(const Grid<3> &grid,
                   const zisa::array_const_view<complex_t, 4> &u_hat) {
  return spectrum<EnstrophySpectrumOperator>(grid, u_hat);
}

#if AZEBAN_HAS_MPI
struct EnstrophySpectrumOperatorMPI {
  static real_t eval(const Grid<1> &, long, complex_t) { return 0; }

  static real_t
  eval(const Grid<2> &grid, long k1, long k2, complex_t u, complex_t v) {
    const complex_t curl_z
        = 2 * zisa::pi * (complex_t(0, k2) * v - complex_t(0, k1) * u);
    const real_t norm = 1. / zisa::pow<2>(grid.N_phys);
    return 0.5 * abs2(norm * curl_z);
  }

  static real_t eval(const Grid<3> &grid,
                     long k1,
                     long k2,
                     long k3,
                     complex_t u,
                     complex_t v,
                     complex_t w) {
    const complex_t curl_x
        = 2 * zisa::pi * (complex_t(0, k2) * w - complex_t(0, k1) * v);
    const complex_t curl_y
        = 2 * zisa::pi * (complex_t(0, k1) * u - complex_t(0, k3) * w);
    const complex_t curl_z
        = 2 * zisa::pi * (complex_t(0, k3) * v - complex_t(0, k2) * u);
    const real_t norm = 1. / zisa::pow<3>(grid.N_phys);
    return 0.5
           * (abs2(norm * curl_x) + abs2(norm * curl_y) + abs2(norm * curl_z));
  }
};

std::vector<real_t>
enstrophy_spectrum(const Grid<1> &grid,
                   const zisa::array_const_view<complex_t, 2> &u_hat,
                   MPI_Comm comm) {
  return spectrum<EnstrophySpectrumOperatorMPI>(grid, u_hat, comm);
}

std::vector<real_t>
enstrophy_spectrum(const Grid<2> &grid,
                   const zisa::array_const_view<complex_t, 3> &u_hat,
                   MPI_Comm comm) {
  return spectrum<EnstrophySpectrumOperatorMPI>(grid, u_hat, comm);
}

std::vector<real_t>
enstrophy_spectrum(const Grid<3> &grid,
                   const zisa::array_const_view<complex_t, 4> &u_hat,
                   MPI_Comm comm) {
  return spectrum<EnstrophySpectrumOperatorMPI>(grid, u_hat, comm);
}
#endif

}
