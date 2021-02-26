#ifndef EQUATION_H_
#define EQUATION_H_

#include <zisa/memory/array_view.hpp>



namespace azeban {


template<typename Scalar, int Dim>
class Equation {
public:
  using scalar_t = Scalar;
  static constexpr int dim_v = Dim;

  Equation() = default;
  Equation(const Equation&) = default;
  Equation(Equation&&) = default;
  virtual ~Equation() = default;
  Equation& operator=(const Equation&) = default;
  Equation& operator=(Equation&&) = default;

  // Replaces the contents of u with its time derivative
  virtual void dudt(const zisa::array_view<scalar_t, dim_v> &u) = 0;
};


}



#endif
