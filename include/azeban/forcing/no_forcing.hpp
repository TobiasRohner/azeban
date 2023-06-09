#ifndef AZEBAN_FORCING_NO_FORCING_HPP_
#define AZEBAN_FORCING_NO_FORCING_HPP_

namespace azeban {

class NoForcing {
public:
  ANY_DEVICE_INLINE void pre(real_t, real_t) {}

  ANY_DEVICE_INLINE void operator()(real_t, long, complex_t *f1) { *f1 = 0; }

  ANY_DEVICE_INLINE void
  operator()(real_t, real_t, long, long, complex_t *f1, complex_t *f2) {
    *f1 = 0;
    *f2 = 0;
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
};

}

#endif
