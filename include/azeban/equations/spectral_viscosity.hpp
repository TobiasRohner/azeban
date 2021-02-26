#ifndef SPECTRAL_VISCOSITY_H_
#define SPECTRAL_VISCOSITY_H_

#include <zisa/config.hpp>
#include <azeban/config.hpp>



namespace azeban {


template<typename Derived>
struct SpectralViscosityBase {
  SpectralViscosityBase(real_t _eps) : eps(_eps) { }

  ANY_DEVICE_INLINE real_t eval(int k) const {
    return eps * k * k * static_cast<const Derived&>(*this).Qk(k);
  }

  real_t eps;
};


struct Step1D : public SpectralViscosityBase<Step1D>{
  using super = SpectralViscosityBase<Step1D>;

  Step1D(real_t _eps, int _mN) : super(_eps), mN(_mN) { }

  ANY_DEVICE_INLINE real_t Qk(int k) const {
    return k > mN ? 1 : 0;
  }

  using super::eval;

  using super::eps;
  int mN;
};


}



#endif
