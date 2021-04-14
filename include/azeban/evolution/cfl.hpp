#ifndef CFL_H_
#define CFL_H_

#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <azeban/operations/norm.hpp>
#include <azeban/profiler.hpp>
#include <zisa/memory/array_view.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi_types.hpp>
#include <mpi.h>
#endif

namespace azeban {

template <int Dim>
class CFL {
public:
  static constexpr int dim_v = Dim;

  CFL(const Grid<Dim> &grid, real_t C) : grid_(grid), C_(C) {}
  CFL() = default;
  CFL(const CFL &) = default;
  CFL(CFL &&) = default;
  ~CFL() = default;

  CFL &operator=(const CFL &) = default;
  CFL &operator=(CFL &&) = default;

  real_t dt(const zisa::array_const_view<complex_t, dim_v + 1> &u_hat) const {
    AZEBAN_PROFILE_START("CFL::dt");
    const real_t sup = norm(u_hat, 1);
    AZEBAN_PROFILE_STOP("CFL::dt");
    return zisa::pow<dim_v - 1>(grid_.N_phys) * C_ / sup;
  }

#if AZEBAN_HAS_MPI
  real_t dt(const zisa::array_const_view<complex_t, dim_v + 1> &u_hat,
            MPI_Comm comm) const {
    AZEBAN_PROFILE_START("CFL::dt");
    real_t sup = norm(u_hat, 1);
    MPI_Allreduce(&sup, MPI_IN_PLACE, 1, mpi_type(sup), MPI_SUM, comm);
    AZEBAN_PROFILE_STOP("CFL::dt");
    return zisa::pow<dim_v - 1>(grid_.N_phys) * C_ / sup;
  }
#endif

  const Grid<dim_v> &grid() const { return grid_; }

private:
  Grid<dim_v> grid_;
  real_t C_;
};

}

#endif
