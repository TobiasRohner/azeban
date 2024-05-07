#ifndef AZEBAN_FORCING_BOUSSINESQ_HPP_
#define AZEBAN_FORCING_BOUSSINESQ_HPP_

#include <zisa/math/basic_functions.hpp>

namespace azeban {

class Boussinesq {
public:
  ANY_DEVICE_INLINE Boussinesq() {}

  ANY_DEVICE_INLINE void pre(real_t, real_t) {}

  ANY_DEVICE_INLINE void
  operator()(real_t, complex_t, complex_t, long, complex_t *f1) {
    *f1 = 0;
  }

  ANY_DEVICE_INLINE void operator()(real_t,
                                    real_t,
                                    complex_t,
                                    complex_t,
                                    complex_t rho,
                                    long,
                                    long,
                                    complex_t *f1,
                                    complex_t *f2) {
    *f1 = 0;
    *f2 = rho;
  }

  ANY_DEVICE_INLINE void operator()(real_t,
                                    real_t,
                                    complex_t,
                                    complex_t,
                                    complex_t,
                                    complex_t rho,
                                    int,
                                    int,
                                    int,
                                    complex_t *f1,
                                    complex_t *f2,
                                    complex_t *f3) {
    *f1 = 0;
    *f2 = 0;
    *f3 = rho;
  }

  void destroy() {}
};

}

#endif
