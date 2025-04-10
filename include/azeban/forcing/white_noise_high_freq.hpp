#ifndef AZEBAN_FORCING_WHITE_NOISE_HIGH_FREQ_HPP_
#define AZEBAN_FORCING_WHITE_NOISE_HIGH_FREQ_HPP_

#include <azeban/grid.hpp>
#include <azeban/logging.hpp>
#include <azeban/random/rng_traits.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/cuda_check_error.hpp>
#include <azeban/cuda/forcing/white_noise_high_freq_cuda.hpp>
#include <azeban/cuda/random/curand_helpers.hpp>
#endif
#include <numeric>
#include <omp.h>
#include <vector>

namespace azeban {

template <int Dim,
          typename RNG,
          zisa::device_type LOCATION = RNGTraits<RNG>::location>
class WhiteNoiseHighFreq {
  static_assert(LOCATION != LOCATION, "");
};

#ifndef __NVCC__
template <typename RNG>
class WhiteNoiseHighFreq<1, RNG, zisa::device_type::cpu> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  explicit WhiteNoiseHighFreq(
      const Grid<1> &, real_t, int, int, real_t, unsigned long long) {}

  void pre(real_t, real_t) {}

  void
  operator()(real_t, real_t dt, complex_t, complex_t, int k1, complex_t *f1) {
    *f1 = 0;
  }
};

template <typename RNG>
class WhiteNoiseHighFreq<2, RNG, zisa::device_type::cpu> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  explicit WhiteNoiseHighFreq(const Grid<2> &grid,
                              real_t b,
                              int k_min,
                              int k_max,
                              real_t eps,
                              unsigned long long seed)
      : b_(zisa::sqrt(b
                      / ((2 * k_max - 1) * (2 * k_max - 1)
                         - (2 * k_min - 1) * (2 * k_min - 1)))),
        k_min_(k_min),
        k_max_(k_max),
        eps_(eps),
        state_(seed),
        dist_(0, zisa::sqrt(zisa::pow<2>(static_cast<real_t>(grid.N_phys)))),
        pot_(zisa::shape_t<2>(2 * (k_max + k_min + 2), 2 * (k_max - k_min))) {}

  void pre(real_t, real_t) {
    for (size_t i = 0; i < pot_.shape(0); ++i) {
      for (size_t j = 0; j < pot_.shape(1); ++j) {
        pot_(i, j) = dist_(state_);
      }
    }
  }

  void operator()(real_t,
                  real_t dt,
                  complex_t,
                  complex_t,
                  complex_t,
                  int k1,
                  int k2,
                  complex_t *f1,
                  complex_t *f2) {
    const int absk1 = zisa::abs(k1);
    const int absk2 = zisa::abs(k2);
    if ((absk1 < k_min_ && absk2 < k_min_) || absk1 >= k_max_
        || absk2 >= k_max_) {
      *f1 = 0;
      *f2 = 0;
    } else {
      const bool inZp = (k1 + (k2 > 0)) > 0;
      const int k1p = inZp ? k1 : -k1;
      const int k2p = inZp ? k2 : -k2;
      unsigned idx1, idx2;
      if (k1p >= k_min_) {
        idx1 = k2p + k_max_ - 1;
        idx2 = k1p - k_min_;
      } else {
        if (k2p > 0) {
          idx1 = k2p + k_max_ + k1p;
          idx2 = k2p - k_min_;
        } else {
          idx1 = k2p + k_max_ + k_min_ + k1p - 1;
          idx2 = -k2p - k_min_;
        }
      }
      const real_t knorm
          = 2 * zisa::pi * zisa::sqrt(static_cast<real_t>(k1 * k1 + k2 * k2));
      const real_t coeff
          = b_ * zisa::sqrt(eps_ / dt / zisa::sqrt(zisa::pi)) / knorm;
      const real_t eta = pot_(idx1, idx2);
      const real_t nu = pot_(idx1, idx2 + (k_max_ - k_min_));
      if (inZp) {
        *f1 = coeff * complex_t(k2 * eta, k2 * nu);
        *f2 = coeff * complex_t(-k1 * eta, -k1 * nu);
      } else {
        *f1 = coeff * complex_t(k2 * eta, k2 * nu);
        *f2 = coeff * complex_t(-k1 * eta, k1 * nu);
      }
    }
  }

private:
  real_t b_;
  int k_min_;
  int k_max_;
  real_t eps_;
  state_t state_;
  std::normal_distribution<real_t> dist_;
  zisa::array<real_t, 2> pot_;
};

template <typename RNG>
class WhiteNoiseHighFreq<3, RNG, zisa::device_type::cpu> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  explicit WhiteNoiseHighFreq(
      const Grid<3> &, real_t, int, int, real_t, unsigned long long) {}

  void pre(real_t, real_t) {}

  void operator()(real_t,
                  real_t,
                  complex_t,
                  complex_t,
                  complex_t,
                  complex_t,
                  int,
                  int,
                  int,
                  complex_t *f1,
                  complex_t *f2,
                  complex_t *f3) {
    *f1 = 0;
    *f2 = 0;
    *f3 = 0;
  }
};

#else
template <int Dim, typename RNG>
class WhiteNoiseHighFreq<Dim, RNG, zisa::device_type::cpu> {
public:
  explicit WhiteNoiseHighFreq(
      const Grid<Dim> &, real_t, int, int, real_t, unsigned long long) {}
  __device__ void pre(real_t, real_t) {}
  __device__ void
  operator()(real_t, real_t, complex_t, complex_t, int, complex_t *f1) {}
  __device__ void operator()(real_t,
                             real_t,
                             complex_t,
                             complex_t,
                             complex_t,
                             int,
                             int,
                             complex_t *f1,
                             complex_t *f2) {}
  __device__ void operator()(real_t,
                             real_t,
                             complex_t,
                             complex_t,
                             complex_t,
                             complex_t,
                             int,
                             int,
                             int,
                             complex_t *f1,
                             complex_t *f2,
                             complex_t *f3) {}
};
#endif

#if ZISA_HAS_CUDA

template <typename RNG>
class WhiteNoiseHighFreq<1, RNG, zisa::device_type::cuda> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  explicit WhiteNoiseHighFreq(
      const Grid<1> &, real_t, int, int, real_t, unsigned long long) {}

  void destroy() {}

  __device__ __inline__ void pre(real_t, real_t) {}

  __device__ __inline__ void
  operator()(real_t t, real_t dt, complex_t, complex_t, int k1, complex_t *f1) {
    f1->x = 0;
    f1->y = 0;
  }
};

template <typename RNG>
class WhiteNoiseHighFreq<2, RNG, zisa::device_type::cuda> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  explicit WhiteNoiseHighFreq(const Grid<2> &grid,
                              real_t b,
                              int k_min,
                              int k_max,
                              real_t eps,
                              unsigned long long seed)
      : grid_(grid),
        b_(zisa::sqrt(b
                      / ((2 * k_max - 1) * (2 * k_max - 1)
                         - (2 * k_min - 1) * (2 * k_min - 1)))),
        k_min_(k_min),
        k_max_(k_max),
        eps_(eps),
        pot_(zisa::shape_t<2>(0, 0), nullptr, zisa::device_type::cuda) {
    zisa::shape_t<2> shape(2 * (k_max + k_min + 2), 2 * (k_max - k_min));
    real_t *data;
    const auto err = cudaMalloc(&data, zisa::product(shape) * sizeof(real_t));
    cudaCheckError(err);
    pot_ = zisa::array_view<real_t, 2>(shape, data, zisa::device_type::cuda);
    const size_t Ns = pot_.size();
    curand_allocate_state<RNG>(&state_, Ns, seed);
  }
  WhiteNoiseHighFreq(const WhiteNoiseHighFreq &) = default;
  WhiteNoiseHighFreq(WhiteNoiseHighFreq &&) = default;

  ~WhiteNoiseHighFreq() = default;

  WhiteNoiseHighFreq &operator=(const WhiteNoiseHighFreq &) = default;
  WhiteNoiseHighFreq &operator=(WhiteNoiseHighFreq &&) = default;

  void destroy() {
    curand_free_state<RNG>(state_);
    cudaFree(pot_.raw());
  }

  void pre(real_t, real_t) { white_noise_high_freq_pre_cuda(pot_, state_); }

  __device__ __inline__ void operator()(real_t,
                                        real_t dt,
                                        complex_t,
                                        complex_t,
                                        complex_t,
                                        int k1,
                                        int k2,
                                        complex_t *f1,
                                        complex_t *f2) {
    const real_t norm = zisa::pow<2>(static_cast<real_t>(grid_.N_phys));
    const int absk1 = zisa::abs(k1);
    const int absk2 = zisa::abs(k2);
    if ((absk1 < k_min_ && absk2 < k_min_) || absk1 >= k_max_
        || absk2 >= k_max_) {
      *f1 = 0;
      *f2 = 0;
    } else {
      const bool inZp = (k1 + (k2 > 0)) > 0;
      const int k1p = inZp ? k1 : -k1;
      const int k2p = inZp ? k2 : -k2;
      unsigned idx1, idx2;
      if (k1p >= k_min_) {
        idx1 = k2p + k_max_ - 1;
        idx2 = k1p - k_min_;
      } else {
        if (k2p > 0) {
          idx1 = k2p + k_max_ + k1p;
          idx2 = k2p - k_min_;
        } else {
          idx1 = k2p + k_max_ + k_min_ + k1p - 1;
          idx2 = -k2p - k_min_;
        }
      }
      const real_t knorm
          = 2 * zisa::pi * zisa::sqrt(static_cast<real_t>(k1 * k1 + k2 * k2));
      const real_t coeff
          = norm * b_ * zisa::sqrt(eps_ / dt / zisa::sqrt(zisa::pi)) / knorm;
      const real_t eta = pot_(idx1, idx2);
      const real_t nu = pot_(idx1, idx2 + (k_max_ - k_min_));
      if (inZp) {
        *f1 = coeff * complex_t(k2 * eta, k2 * nu);
        *f2 = coeff * complex_t(-k1 * eta, -k1 * nu);
      } else {
        *f1 = coeff * complex_t(k2 * eta, k2 * nu);
        *f2 = coeff * complex_t(-k1 * eta, k1 * nu);
      }
    }
  }

private:
  Grid<2> grid_;
  real_t b_;
  int k_min_;
  int k_max_;
  real_t eps_;
  state_t *state_;
  zisa::array_view<real_t, 2> pot_;
};

template <typename RNG>
class WhiteNoiseHighFreq<3, RNG, zisa::device_type::cuda> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  explicit WhiteNoiseHighFreq(
      const Grid<3> &, real_t, int, int, real_t, unsigned long long) {}

  void destroy() {}

  void pre(real_t, real_t) {}

  __device__ __inline__ void operator()(real_t,
                                        real_t,
                                        complex_t,
                                        complex_t,
                                        complex_t,
                                        complex_t,
                                        int,
                                        int,
                                        int,
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
