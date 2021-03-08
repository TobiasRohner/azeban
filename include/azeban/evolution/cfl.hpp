#ifndef CFL_H_
#define CFL_H_

#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <azeban/operations/norm.hpp>
#include <zisa/memory/array_view.hpp>

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
    const real_t sup = norm(u_hat, 1);
    return std::pow(grid_.N_phys, dim_v - 1) * C_ / sup;
  }

  const Grid<dim_v> &grid() const { return grid_; }

private:
  Grid<dim_v> grid_;
  real_t C_;
};

}

#endif
