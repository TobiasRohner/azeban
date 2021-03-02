#ifndef CFL_H_
#define CFL_H_

#include <azeban/config.hpp>
#include <azeban/operations/norm.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

class CFL {
public:
  CFL(real_t C) : C_(C) {}
  CFL() = default;
  CFL(const CFL &) = default;
  CFL(CFL &&) = default;
  ~CFL() = default;

  CFL &operator=(const CFL &) = default;
  CFL &operator=(CFL &&) = default;

  template <int Dim>
  real_t dt(const zisa::array_const_view<complex_t, Dim> &u_hat) const {
    const real_t sup = norm(u_hat, 1);
    return C_ / sup;
  }

private:
  real_t C_;
};

}

#endif
