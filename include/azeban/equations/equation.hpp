#ifndef EQUATION_H_
#define EQUATION_H_

#include <azeban/grid.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename Scalar, int Dim>
class Equation {
public:
  using scalar_t = Scalar;
  static constexpr int dim_v = Dim;

  Equation() = delete;
  Equation(const Grid<Dim> &grid) : grid_(grid) {}
  Equation(const Equation &) = default;
  Equation(Equation &&) = default;
  virtual ~Equation() = default;
  Equation &operator=(const Equation &) = default;
  Equation &operator=(Equation &&) = default;

  // Replaces the contents of u with its time derivative
  virtual void dudt(const zisa::array_view<scalar_t, dim_v + 1> &u) = 0;

  const Grid<Dim> &grid() const { return grid_; }

protected:
  Grid<Dim> grid_;
};

}

#endif
