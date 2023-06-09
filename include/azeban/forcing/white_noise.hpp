#ifndef AZEBAN_FORCING_WHITE_NOISE_HPP_
#define AZEBAN_FORCING_WHITE_NOISE_HPP_

#include <azeban/grid.hpp>
#include <azeban/logging.hpp>
#include <azeban/random/rng_traits.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/forcing/white_noise_cuda.hpp>
#include <azeban/cuda/random/curand_helpers.hpp>
#endif
#include <numeric>
#include <omp.h>
#include <vector>

namespace azeban {

template <int Dim,
          typename RNG,
          zisa::device_type LOCATION = RNGTraits<RNG>::location>
class WhiteNoise {
  static_assert(LOCATION != LOCATION, "");
};

template <typename RNG>
class WhiteNoise<1, RNG, zisa::device_type::cpu> {
  using state_t = typename RNGTraits<RNG>::state_t;

#ifdef __NVCC__
public:
  __device__ void pre(real_t, real_t) {}
  __device__ void operator()(real_t, real_t, int, complex_t *f1) {}
#else
public:
  explicit WhiteNoise(const Grid<1> &, real_t, int, unsigned long long) {}

  void pre(real_t, real_t) {}

  void operator()(real_t, real_t dt, int k1, complex_t *f1) { *f1 = 0; }
#endif
};

template <typename RNG>
class WhiteNoise<2, RNG, zisa::device_type::cpu> {
  using state_t = typename RNGTraits<RNG>::state_t;

#ifdef __NVCC__
public:
  __device__ void pre(real_t, real_t) {}
  __device__ void
  operator()(real_t, real_t, int, int, complex_t *, complex_t *) {}
#else
public:
  explicit WhiteNoise(const Grid<2> &grid,
                      real_t sigma,
                      int N,
                      unsigned long long seed)
      : state_(seed),
        dist_(0, zisa::pow<2>(static_cast<real_t>(grid.N_phys)) * sigma),
        pot_(zisa::shape_t<2>(N, N)) {}

  void pre(real_t, real_t) {
    for (size_t i = 0; i < pot_.shape(0); ++i) {
      for (size_t j = 0; j < pot_.shape(1); ++j) {
        pot_(i, j) = i > 0 && j > 0 ? dist_(state_) : 0;
      }
    }
  }

  void
  operator()(real_t, real_t dt, int k1, int k2, complex_t *f1, complex_t *f2) {
    const unsigned absk1 = zisa::abs(k1);
    const unsigned absk2 = zisa::abs(k2);
    const real_t knorm = zisa::sqrt(static_cast<real_t>(k1 * k1 + k2 * k2));
    const int s1 = k1 >= 0 ? 1 : -1;
    const int s2 = k2 >= 0 ? 1 : -1;
    if (absk1 < pot_.shape(0) && absk2 < pot_.shape(1)) {
      const real_t coeff = 2. / knorm * pot_(absk1, absk2) / zisa::sqrt(dt);
      *f1 = complex_t(coeff * absk2 / 4., 0);
      *f2 = complex_t(-coeff * absk1 / 4. * s1 * s2, 0);
    } else {
      *f1 = 0;
      *f2 = 0;
    }
  }
#endif
private:
#ifndef __NVCC__
  state_t state_;
  std::normal_distribution<real_t> dist_;
  zisa::array<real_t, 2> pot_;
#endif
};

template <typename RNG>
class WhiteNoise<3, RNG, zisa::device_type::cpu> {
  using state_t = typename RNGTraits<RNG>::state_t;

#ifdef __NVCC__
public:
  __device__ void pre(real_t, real_t) {}
  __device__ void operator()(
      real_t, real_t, int, int, int, complex_t *, complex_t *, complex_t *) {}
#else
public:
  explicit WhiteNoise(const Grid<3> &grid,
                      real_t sigma,
                      int N,
                      unsigned long long seed)
      : state_(seed),
        dist_(0, zisa::pow<3>(static_cast<real_t>(grid.N_phys)) * sigma),
        pot_(zisa::shape_t<3>(N, N, N)) {}

  void pre(real_t, real_t) {
    for (size_t i = 0; i < pot_.size(); ++i) {
      pot_[i] = dist_(state_);
    }
  }

  void operator()(real_t,
                  real_t dt,
                  int k1,
                  int k2,
                  int k3,
                  complex_t *f1,
                  complex_t *f2,
                  complex_t *f3) {
    AZEBAN_ERR("Not yet Implemented");
    *f1 = 0;
    *f2 = 0;
    *f3 = 0;
  }
#endif
private:
#ifndef __NVCC__
  state_t state_;
  std::normal_distribution<real_t> dist_;
  zisa::array<real_t, 3> pot_;
#endif
};

#if ZISA_HAS_CUDA

template <typename RNG>
class WhiteNoise<1, RNG, zisa::device_type::cuda> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  template <int Dim>
  explicit WhiteNoise(const Grid<Dim> &grid,
                      real_t sigma,
                      int N,
                      unsigned long long seed) {}
  WhiteNoise(const WhiteNoise &) = default;
  WhiteNoise(WhiteNoise &&) = default;

  ~WhiteNoise() = default;

  WhiteNoise &operator=(const WhiteNoise &) = default;
  WhiteNoise &operator=(WhiteNoise &&) = default;

  void destroy() {}

  __device__ __inline__ void pre(real_t, real_t) {}

  __device__ __inline__ void
  operator()(real_t t, real_t dt, int k1, complex_t *f1) {
    f1->x = 0;
    f1->y = 0;
  }
};

template <typename RNG>
class WhiteNoise<2, RNG, zisa::device_type::cuda> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  explicit WhiteNoise(const Grid<2> &grid,
                      real_t sigma,
                      int N,
                      unsigned long long seed)
      : sigma_(zisa::pow<2>(static_cast<real_t>(grid.N_phys)) * sigma),
        pot_(zisa::shape_t<2>(N, N), zisa::device_type::cuda) {
    const size_t Ns = zisa::pow<2>(N);
    curand_allocate_state<RNG>(&state_, Ns, seed);
  }
  WhiteNoise(const WhiteNoise &) = default;
  WhiteNoise(WhiteNoise &&) = default;

  ~WhiteNoise() = default;

  WhiteNoise &operator=(const WhiteNoise &) = default;
  WhiteNoise &operator=(WhiteNoise &&) = default;

  void destroy() { curand_free_state<RNG>(state_); }

  void pre(real_t, real_t) { white_noise_pre_cuda(sigma_, pot_, state_); }

  __device__ __inline__ void
  operator()(real_t, real_t dt, int k1, int k2, complex_t *f1, complex_t *f2) {
    const unsigned absk1 = zisa::abs(k1);
    const unsigned absk2 = zisa::abs(k2);
    const real_t knorm = zisa::sqrt(static_cast<real_t>(k1 * k1 + k2 * k2));
    const int s1 = k1 >= 0 ? 1 : -1;
    const int s2 = k2 >= 0 ? 1 : -1;
    if (absk1 < pot_.shape(0) && absk2 < pot_.shape(1)) {
      const real_t coeff = 2. / knorm * pot_(absk1, absk2) / zisa::sqrt(dt);
      *f1 = complex_t(coeff * absk2 / 4., 0);
      *f2 = complex_t(-coeff * absk1 / 4. * s1 * s2, 0);
    } else {
      *f1 = 0;
      *f2 = 0;
    }
  }

private:
  state_t *state_;
  real_t sigma_;
  zisa::array<real_t, 2> pot_;
};

template <typename RNG>
class WhiteNoise<3, RNG, zisa::device_type::cuda> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  explicit WhiteNoise(const Grid<3> &grid,
                      real_t sigma,
                      int N,
                      unsigned long long seed) {}
  WhiteNoise(const WhiteNoise &) = default;
  WhiteNoise(WhiteNoise &&) = default;

  ~WhiteNoise() = default;

  WhiteNoise &operator=(const WhiteNoise &) = default;
  WhiteNoise &operator=(WhiteNoise &&) = default;

  void destroy() {}

  __device__ __inline__ void pre(real_t, real_t) {}

  __device__ __inline__ void operator()(real_t t,
                                        real_t dt,
                                        int k1,
                                        int k2,
                                        int k3,
                                        complex_t *f1,
                                        complex_t *f2,
                                        complex_t *f3) {
    *f1 = 0;
    *f2 = 0;
    *f3 = 0;
  }
};

#endif

}

#endif
