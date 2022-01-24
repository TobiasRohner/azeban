#ifndef AZEBAN_FORCING_WHITE_NOISE_HPP_
#define AZEBAN_FORCING_WHITE_NOISE_HPP_

#include <azeban/random/rng_traits.hpp>
#include <azeban/grid.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/random/curand_helpers.hpp>
#endif


namespace azeban {

template<int Dim, typename RNG, zisa::device_type LOCATION = RNGTraits<RNG>::location>
class WhiteNoise {
  static_assert(LOCATION != LOCATION, "");
};

template<int Dim, typename RNG>
class WhiteNoise<Dim, RNG, zisa::device_type::cpu> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  explicit WhiteNoise(const Grid<Dim> &grid, real_t sigma, unsigned long long seed) : state_(seed), dist_(0, zisa::sqrt(zisa::pow<Dim>(static_cast<real_t>(grid.N_phys)/2))*sigma) { }
  WhiteNoise(const WhiteNoise &) = delete;
  WhiteNoise(WhiteNoise &&) = default;

  ~WhiteNoise() = default;

  WhiteNoise &operator=(const WhiteNoise &) = delete;
  WhiteNoise &operator=(WhiteNoise &&) = default;

  complex_t operator()(zisa::int_t) { return generate(); }
  complex_t operator()(zisa::int_t, zisa::int_t) { return generate(); }
  complex_t operator()(zisa::int_t, zisa::int_t, zisa::int_t) { return generate(); }

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

template<int Dim, typename RNG>
class WhiteNoise<Dim, RNG, zisa::device_type::cuda> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  explicit WhiteNoise(const Grid<Dim> &grid, real_t sigma, unsigned long long seed) : sigma_(zisa::sqrt(zisa::pow(static_cast<real_t>(grid.N_phys)/2))*sigma) {
    const size_t N = zisa::pow<Dim-1>(grid.N_phys) * grid.N_fourier;
    curand_allocate_state(state_, N, seed);
  }
  WhiteNoise(const WhiteNoise &) = delete;
  WhiteNoise(WhiteNoise &&) = default;

  ~WhiteNoise() {
    curand_free_state(state_);
  }

  WhiteNoise &operator=(const WhiteNoise &) = delete;
  WhiteNoise &operator=(WhiteNoise &&) = default;

  __device__ __inline__ real_t operator()(zisa::int_t) { return generate(); }
  __device__ __inline__ real_t operator()(zisa::int_t, zisa::int_t) { return generate(); }
  __device__ __inline__ real_t operator()(zisa::int_t, zisa::int_t, zisa::int_t) { return generate(); }

private:
  state_t state_;
  real_t sigma_;

  __device__ __inline__ real_t generate() {
    const size_t id = gridDim.y*gridDim.z*blockDim.x*blockDim.y*threadIdx.x + gridDim.z*blockDim.z*threadIdx.y + threadIdx.z;
    const auto norm = normal2(&state_[id], real_t{});
    complex_t res;
    res.x = norm.x;
    res.y = norm.y;
    return res;
  }

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
