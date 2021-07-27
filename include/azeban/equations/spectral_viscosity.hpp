#ifndef SPECTRAL_VISCOSITY_H_
#define SPECTRAL_VISCOSITY_H_

#include <azeban/config.hpp>
#include <zisa/config.hpp>
#include <zisa/math/basic_functions.hpp>

namespace azeban {

template <typename Derived>
struct SpectralViscosityBase {
  SpectralViscosityBase(real_t _eps) : eps(_eps) {}

  ANY_DEVICE_INLINE real_t eval(real_t k) const {
    return -eps * k * k * static_cast<const Derived &>(*this).Qk(k);
  }

  real_t eps;
};

struct Step1D final : public SpectralViscosityBase<Step1D> {
  using super = SpectralViscosityBase<Step1D>;

  Step1D(real_t _eps, real_t _k0) : super(_eps), k0(_k0) {}

  ANY_DEVICE_INLINE real_t Qk(real_t k) const {
    return zisa::abs(k) > k0 ? 1 : 0;
  }

  using super::eval;

  using super::eps;
  real_t k0;
};

struct SmoothCutoff1D final : public SpectralViscosityBase<SmoothCutoff1D> {
  using super = SpectralViscosityBase<SmoothCutoff1D>;

  SmoothCutoff1D(real_t _eps, real_t _k0) : super(_eps), k0(_k0) {}

  ANY_DEVICE_INLINE real_t Qk(real_t k) const {
    const real_t k1 = zisa::abs(k) / k0;
    const real_t k2 = k1 * k1;
    const real_t k4 = k2 * k2;
    const real_t k8 = k4 * k4;
    const real_t k16 = k8 * k8;
    const real_t k18 = k16 * k2;
    return 1 - zisa::exp(-k18);
  }

  using super::eval;

  using super::eps;
  real_t k0;
};

/*
struct Quadratic final : public SpectralViscosityBase<Quadratic> {
  using super = SpectralViscosityBase<Quadratic>;

  Quadratic(real_t _eps, zisa::int_t _N_phys) : super(_eps), N(_N_phys) {}

  ANY_DEVICE_INLINE real_t Qk(real_t k) const {
    const real_t sqrtN = zisa::sqrt(N);
    const real_t absk = zisa::abs(k / (2 * zisa::pi));
    if (absk >= sqrtN) {
      return 1. - static_cast<real_t>(N) / (absk * absk);
    }
    else {
      return 0;
    }
  }

  using super::eval;

  using super::eps;
  zisa::int_t N;
};
*/

struct Quadratic final {
  Quadratic(real_t _eps, zisa::int_t _N_phys)
      : eps(_eps / _N_phys), N(_N_phys) {}

  ANY_DEVICE_INLINE real_t Qk(real_t k) const {
    const real_t sqrtN = zisa::sqrt(N);
    const real_t absk = zisa::abs(k / (2 * zisa::pi));
    if (absk >= sqrtN) {
      return 1. - static_cast<real_t>(N) / (absk * absk);
    } else {
      return 0;
    }
  }

  ANY_DEVICE_INLINE real_t eval(real_t k) const {
    const real_t knorm = k / (2 * zisa::pi);
    return -eps * zisa::max(real_t(0), knorm * knorm - N);
  }

  real_t eps;
  zisa::int_t N;
};

struct NoViscosity final {
  ANY_DEVICE_INLINE real_t Qk(real_t /* k */) const { return 0.0; }
  ANY_DEVICE_INLINE real_t eval(real_t /* k */) const { return 0.0; }
};

}

#endif
