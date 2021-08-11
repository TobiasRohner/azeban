#ifndef EQUATION_H_
#define EQUATION_H_

#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
class Equation {
public:
  static constexpr int dim_v = Dim;

  Equation(const Grid<Dim> &grid) : grid_(grid){};
  Equation(const Equation &) = default;
  Equation(Equation &&) = default;
  virtual ~Equation() = default;
  Equation &operator=(const Equation &) = default;
  Equation &operator=(Equation &&) = default;

  // Replaces the contents of u with its time derivative
  virtual void dudt(const zisa::array_view<complex_t, dim_v + 1> &u) = 0;

  // const Grid<Dim> &grid() const { return grid_; }
  virtual int n_vars() const = 0;

  virtual void *get_fft_work_area() const { return nullptr; }

protected:
  Grid<Dim> grid_;
};

}

#endif
