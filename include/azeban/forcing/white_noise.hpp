#ifndef AZEBAN_FORCING_WHITE_NOISE_HPP_
#define AZEBAN_FORCING_WHITE_NOISE_HPP_

#include <azeban/grid.hpp>
#include <azeban/random/rng_traits.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/random/curand_helpers.hpp>
#endif

namespace azeban {

template <int Dim,
          typename RNG,
          zisa::device_type LOCATION = RNGTraits<RNG>::location>
class WhiteNoise {
  static_assert(LOCATION != LOCATION, "");
};

template <int Dim, typename RNG>
class WhiteNoise<Dim, RNG, zisa::device_type::cpu> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  explicit WhiteNoise(const Grid<Dim> &grid,
                      real_t sigma,
                      unsigned long long seed)
      : state_(seed),
        dist_(0,
              zisa::sqrt(zisa::pow<Dim>(static_cast<real_t>(grid.N_phys) / 2))
                  * sigma) {}
  WhiteNoise(const WhiteNoise &) = delete;
  WhiteNoise(WhiteNoise &&) = default;

  ~WhiteNoise() = default;

  WhiteNoise &operator=(const WhiteNoise &) = delete;
  WhiteNoise &operator=(WhiteNoise &&) = default;

  void operator()(real_t t, zisa::int_t, complex_t *f1) { *f1 = generate(); }

  void
  operator()(real_t t, zisa::int_t, zisa::int_t, complex_t *f1, complex_t *f2) {
    *f1 = generate();
    *f2 = generate();
  }

  void operator()(real_t t,
                  zisa::int_t,
                  zisa::int_t,
                  zisa::int_t,
                  complex_t *f1,
                  complex_t *f1,
                  complex_t *f3) {
    *f1 = generate();
    *f2 = generate();
    *f3 = generate();
  }

private:
  state_t state_;
  std::normal_distribution<real_t> dist_;

  complex_t generate() {
    complex_t res;
    res.x = dist_(state_);
    res.y = dist_(state_);
    return res;
  }
};

#if ZISA_HAS_CUDA

template <int Dim, typename RNG>
class WhiteNoise<Dim, RNG, zisa::device_type::cuda> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  explicit WhiteNoise(const Grid<Dim> &grid,
                      real_t sigma,
                      unsigned long long seed)
      : sigma_(zisa::sqrt(zisa::pow(static_cast<real_t>(grid.N_phys) / 2))
               * sigma) {
    const size_t N = zisa::pow<Dim - 1>(grid.N_phys) * grid.N_fourier;
    curand_allocate_state(state_, N, seed);
  }
  WhiteNoise(const WhiteNoise &) = delete;
  WhiteNoise(WhiteNoise &&) = default;

  ~WhiteNoise() { curand_free_state(state_); }

  WhiteNoise &operator=(const WhiteNoise &) = delete;
  WhiteNoise &operator=(WhiteNoise &&) = default;

  __device__ __inline__ void operator()(real_t t, zisa::int_t, complex_t *f1) {
    const size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    const auto norm = normal2(&state_[id], real_t{});
    f1->x = norm.x;
    f1->y = norm.y;
  }

  __device__ __inline__ void
  operator()(real_t t, zisa::int_t, zisa::int_t, complex_t *f1, complex_t *f2) {
    const size_t idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t id = gridDim.x * blockDim.x * idx_x + idx_y;
    auto local_state = state_[id];
    const auto norm1 = normal2(&local_state, real_t{});
    const auto norm2 = normal2(&local_state, real_t{});
    state_[id] = local_state;
    f1->x = norm1.x;
    f1->y = norm1.y;
    f2->x = norm2.x;
    f2->y = norm2.y;
  }

  __device__ __inline__ void
  operator()(real_t t, zisa::int_t, zisa::int_t, complex_t *f1, complex_t *f2) {
    const size_t idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t idx_z = blockDim.z * blockIdx.z + threadIdx.z;
    const size_t id
        = gridDim.z * blockDim.z * (gridDim.y * blockDim.y * idx_x + idx_y)
          + idx_z;
    auto local_state = state_[id];
    const auto norm1 = normal2(&local_state, real_t{});
    const auto norm2 = normal2(&local_state, real_t{});
    const auto norm3 = normal2(&local_state, real_t{});
    state_[id] = local_state;
    f1->x = norm1.x;
    f1->y = norm1.y;
    f2->x = norm2.x;
    f2->y = norm2.y;
    f3->x = norm3.x;
    f3->y = norm3.y;
  }

private:
  state_t state_;
  real_t sigma_;

  __device__ __inline__ float2 normal2(RNG *rng, float) {
    return curand_normal2(rng);
  }

  __device__ __inline__ double2 normal(RNG *rng, double) {
    return curand_normal2_double(rng);
  }
};

#endif

}

#endif
