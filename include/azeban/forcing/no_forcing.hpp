#ifndef AZEBAN_FORCING_NO_FORCING_HPP_
#define AZEBAN_FORCING_NO_FORCING_HPP_


namespace azeban {

class NoForcing {
public:
  ANY_DEVICE_INLINE real_t operator()(zisa::int_t) { return 0; }
  ANY_DEVICE_INLINE real_t operator()(zisa::int_t, zisa::int_t) { return 0; }
  ANY_DEVICE_INLINE real_t operator()(zisa::int_t, zisa::int_t, zisa::int_t) { return 0; }
};

}


#endif
