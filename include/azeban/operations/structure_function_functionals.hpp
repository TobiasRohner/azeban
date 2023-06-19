#ifndef AZEBAN_OPERATIONS_STRUCTURE_FUNCTION_FUNCTIONALS_HPP_
#define AZEBAN_OPERATIONS_STRUCTURE_FUNCTION_FUNCTIONALS_HPP_

#include <azeban/config.hpp>
#include <azeban/utils/math.hpp>
#include <zisa/config.hpp>

namespace azeban {

class SFCubeFunctional {
public:
  ANY_DEVICE_INLINE SFCubeFunctional(real_t p) : p_(p) {}
  ANY_DEVICE_INLINE SFCubeFunctional(const SFCubeFunctional &) = default;
  ANY_DEVICE_INLINE SFCubeFunctional &operator=(const SFCubeFunctional &)
      = default;

  ANY_DEVICE_INLINE real_t operator()(real_t ui, real_t uj, ssize_t) const {
    return zisa::pow(::azeban::abs(uj - ui), p_);
  }

  ANY_DEVICE_INLINE real_t operator()(
      real_t uij, real_t vij, real_t ukl, real_t vkl, ssize_t, ssize_t) const {
    return zisa::pow(::azeban::abs(ukl - uij), p_)
           + zisa::pow(::azeban::abs(vkl - vij), p_);
  }

  ANY_DEVICE_INLINE real_t operator()(real_t uijk,
                                      real_t vijk,
                                      real_t wijk,
                                      real_t ulmn,
                                      real_t vlmn,
                                      real_t wlmn,
                                      ssize_t,
                                      ssize_t,
                                      ssize_t) const {
    return zisa::pow(::azeban::abs(ulmn - uijk), p_)
           + zisa::pow(::azeban::abs(vlmn - vijk), p_)
           + zisa::pow(::azeban::abs(wlmn - wijk), p_);
  }

private:
  real_t p_;
};

class SFThirdOrderFunctional {
public:
  ANY_DEVICE_INLINE real_t operator()(real_t ui, real_t uj, ssize_t di) const {
    const real_t du = uj - ui;
    const real_t absdu2 = du * du;
    const real_t rabsn = 1. / zisa::sqrt(di * di);
    const real_t nx = rabsn * static_cast<real_t>(di);
    return absdu2 * (du * nx);
  }

  ANY_DEVICE_INLINE real_t operator()(real_t uij,
                                      real_t vij,
                                      real_t ukl,
                                      real_t vkl,
                                      ssize_t di,
                                      ssize_t dj) const {
    const real_t du = ukl - uij;
    const real_t dv = vkl - vij;
    const real_t absdu2 = du * du + dv * dv;
    const real_t rabsn
        = ::azeban::rhypot(static_cast<real_t>(di), static_cast<real_t>(dj));
    const real_t nx = rabsn * static_cast<real_t>(di);
    const real_t ny = rabsn * static_cast<real_t>(dj);
    return absdu2 * (du * nx + dv * ny);
  }

  ANY_DEVICE_INLINE real_t operator()(real_t uijk,
                                      real_t vijk,
                                      real_t wijk,
                                      real_t ulmn,
                                      real_t vlmn,
                                      real_t wlmn,
                                      ssize_t di,
                                      ssize_t dj,
                                      ssize_t dk) const {
    const real_t du = ulmn - uijk;
    const real_t dv = vlmn - vijk;
    const real_t dw = wlmn - wijk;
    const real_t absdu2 = du * du + dv * dv + dw * dw;
    const real_t rabsn = ::azeban::rhypot(static_cast<real_t>(di),
                                          static_cast<real_t>(dj),
                                          static_cast<real_t>(dk));
    const real_t nx = rabsn * static_cast<real_t>(di);
    const real_t ny = rabsn * static_cast<real_t>(dj);
    const real_t nz = rabsn * static_cast<real_t>(dk);
    return absdu2 * (du * nx + dv * ny + dw * nz);
  }
};

class SFLongitudinalFunctional {
public:
  ANY_DEVICE_INLINE SFLongitudinalFunctional(real_t p) : p_(p) {}
  ANY_DEVICE_INLINE SFLongitudinalFunctional(const SFLongitudinalFunctional &)
      = default;
  ANY_DEVICE_INLINE SFLongitudinalFunctional &
  operator=(const SFLongitudinalFunctional &)
      = default;

  ANY_DEVICE_INLINE real_t operator()(real_t ui, real_t uj, ssize_t di) const {
    const real_t du = uj - ui;
    const real_t rabsn = 1. / zisa::sqrt(di * di);
    const real_t nx = rabsn * static_cast<real_t>(di);
    return zisa::pow(du * nx, p_);
  }

  ANY_DEVICE_INLINE real_t operator()(real_t uij,
                                      real_t vij,
                                      real_t ukl,
                                      real_t vkl,
                                      ssize_t di,
                                      ssize_t dj) const {
    const real_t du = ukl - uij;
    const real_t dv = vkl - vij;
    const real_t rabsn
        = ::azeban::rhypot(static_cast<real_t>(di), static_cast<real_t>(dj));
    const real_t nx = rabsn * static_cast<real_t>(di);
    const real_t ny = rabsn * static_cast<real_t>(dj);
    return zisa::pow(du * nx + dv * ny, p_);
  }

  ANY_DEVICE_INLINE real_t operator()(real_t uijk,
                                      real_t vijk,
                                      real_t wijk,
                                      real_t ulmn,
                                      real_t vlmn,
                                      real_t wlmn,
                                      ssize_t di,
                                      ssize_t dj,
                                      ssize_t dk) const {
    const real_t du = ulmn - uijk;
    const real_t dv = vlmn - vijk;
    const real_t dw = wlmn - wijk;
    const real_t rabsn = ::azeban::rhypot(static_cast<real_t>(di),
                                          static_cast<real_t>(dj),
                                          static_cast<real_t>(dk));
    const real_t nx = rabsn * static_cast<real_t>(di);
    const real_t ny = rabsn * static_cast<real_t>(dj);
    const real_t nz = rabsn * static_cast<real_t>(dk);
    return zisa::pow(du * nx + dv * ny + dw * nz, p_);
  }

private:
  real_t p_;
};

}

#endif
