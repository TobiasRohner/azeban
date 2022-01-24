#ifndef AZEBAN_FORCING_NO_FORCING_HPP_
#define AZEBAN_FORCING_NO_FORCING_HPP_

namespace azeban {

class NoForcing {
public:
  ANY_DEVICE_INLINE void operator()(real_t t, zisa::int_t, complex_t *f1) {
    *f1 = 0;
  }

  ANY_DEVICE_INLINE void
  operator()(real_t t, zisa::int_t, zisa::int_t, complex_t *f1, complex_t *f2) {
    *f1 = 0;
    *f2 = 0;
  }

  ANY_DEVICE_INLINE void operator()(real_t t,
                                    zisa::int_t,
                                    zisa::int_t,
                                    zisa::int_t,
                                    complex_t *f1,
                                    complex_t *f2,
                                    complex_t *f3) {
    *f1 = 0;
    *f2 = 0;
    *f3 = 0;
  }
};

}

#endif
