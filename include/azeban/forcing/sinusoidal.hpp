#ifndef AZEBAN_FORCING_SINUSOIDAL_HPP_
#define AZEBAN_FORCING_SINUSOIDAL_HPP_

#include <zisa/math/basic_functions.hpp>

namespace azeban {

class Sinusoidal {
public:
  ANY_DEVICE_INLINE Sinusoidal(zisa::int_t N_phys, real_t amplitude)
      : N_phys_(N_phys), amplitude_(amplitude) {}

  ANY_DEVICE_INLINE void pre(real_t, real_t) {}

  ANY_DEVICE_INLINE void operator()(real_t, long, complex_t *f1) { *f1 = 0; }

  ANY_DEVICE_INLINE void
  operator()(real_t, real_t, long k1, long k2, complex_t *f1, complex_t *f2) {
    const real_t factor = N_phys_ * N_phys_ * amplitude_ / 2;
    if (k1 == 1 && k2 == 1) {
      *f1 = complex_t(factor, 0);
      *f2 = complex_t(0, -factor);
    } else {
      *f1 = 0;
      *f2 = 0;
    }
  }

  ANY_DEVICE_INLINE void operator()(real_t,
                                    real_t,
                                    long,
                                    long,
                                    long,
                                    complex_t *f1,
                                    complex_t *f2,
                                    complex_t *f3) {
    *f1 = 0;
    *f2 = 0;
    *f3 = 0;
  }

  void destroy() {}

private:
  zisa::int_t N_phys_;
  real_t amplitude_;
};

}

#endif
