#ifndef CFL_H_
#define CFL_H_

#include <zisa/memory/array_view.hpp>
#include <azeban/config.hpp>
#include <azeban/operations/norm.hpp>



namespace azeban {


class CFL {
public:
  CFL(real_t C) : C_(C) { }
  CFL() = default;
  CFL(const CFL&) = default;
  CFL(CFL&&) = default;
  ~CFL() = default;

  CFL& operator=(const CFL&) = default;
  CFL& operator=(CFL&&) = default;

  template<int Dim>
  real_t dt(const zisa::array_const_view<complex_t, Dim> &u) const {
    const real_t sup = norm(u, 1);
    return C_ / sup;
  }

private:
  real_t C_;
};


}



#endif
