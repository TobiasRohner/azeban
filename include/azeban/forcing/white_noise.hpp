#ifndef AZEBAN_FORCING_WHITE_NOISE_HPP_
#define AZEBAN_FORCING_WHITE_NOISE_HPP_

#include <azeban/grid.hpp>
#include <azeban/random/rng_traits.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/random/curand_helpers.hpp>
#endif
#include <numeric>
#include <omp.h>
#include <vector>

namespace azeban {

template <typename RNG, zisa::device_type LOCATION = RNGTraits<RNG>::location>
class WhiteNoise {
  static_assert(LOCATION != LOCATION, "");
};

template <typename RNG>
class WhiteNoise<RNG, zisa::device_type::cpu> {
  using state_t = typename RNGTraits<RNG>::state_t;
  struct padded_state_t {
    static constexpr int cacheline_size = 64;
    static constexpr int size_diff
        = cacheline_size - static_cast<int>(sizeof(state_t));
    static constexpr int padding_size = size_diff < 0 ? 0 : size_diff;
    state_t state;
    char pad[padding_size];
  };

#ifdef __NVCC__
public:
  __device__ void operator()(real_t, long, complex_t *f1) {}
  __device__ void operator()(real_t, long, long, complex_t *f1, complex_t *f2) {
  }
  __device__ void operator()(
      real_t, long, long, long, complex_t *f1, complex_t *f2, complex_t *f3) {}
#else
public:
  template <int Dim>
  explicit WhiteNoise(const Grid<Dim> &grid,
                      real_t sigma,
                      unsigned long long seed)
      : state_(),
        dist_(0,
              zisa::sqrt(zisa::pow<Dim>(static_cast<real_t>(grid.N_phys) / 2))
                  * sigma) {
    int n_threads;
#pragma omp parallel
    {
      if (omp_get_thread_num() == 0) {
        n_threads = omp_get_num_threads();
      }
    }
    state_.resize(n_threads);
    std::vector<size_t> idxs(n_threads);
    std::iota(begin(idxs), end(idxs), seed);
    std::seed_seq seq(begin(idxs), end(idxs));
    std::vector<size_t> seeds(n_threads);
    seq.generate(begin(seeds), end(seeds));
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      state_[tid].state.seed(seeds[tid]);
    }
  }
  WhiteNoise(const WhiteNoise &) = default;
  WhiteNoise(WhiteNoise &&) = default;

  ~WhiteNoise() = default;

  WhiteNoise &operator=(const WhiteNoise &) = default;
  WhiteNoise &operator=(WhiteNoise &&) = default;

  void operator()(real_t, long, complex_t *f1) { *f1 = generate(); }

  void operator()(real_t, long, long, complex_t *f1, complex_t *f2) {
    *f1 = generate();
    *f2 = generate();
  }

  void operator()(
      real_t, long, long, long, complex_t *f1, complex_t *f2, complex_t *f3) {
    *f1 = generate();
    *f2 = generate();
    *f3 = generate();
  }

private:
  std::vector<padded_state_t> state_;
  std::normal_distribution<real_t> dist_;

  complex_t generate() {
    state_t &state = state_[omp_get_thread_num()].state;
    complex_t res;
    res.x = dist_(state);
    res.y = dist_(state);
    return res;
  }
#endif
};

#if ZISA_HAS_CUDA

template <typename RNG>
class WhiteNoise<RNG, zisa::device_type::cuda> {
  using state_t = typename RNGTraits<RNG>::state_t;

public:
  template <int Dim>
  explicit WhiteNoise(const Grid<Dim> &grid,
                      real_t sigma,
                      unsigned long long seed)
      : sigma_(zisa::sqrt(zisa::pow<Dim>(static_cast<real_t>(grid.N_phys) / 2))
               * sigma) {
    Nx_ = grid.N_fourier;
    Ny_ = 1;
    Nz_ = 1;
    if (Dim > 1) {
      Nx_ = grid.N_phys;
      Ny_ = grid.N_fourier;
    }
    if (Dim > 2) {
      Ny_ = grid.N_phys;
      Nz_ = grid.N_fourier;
    }
    const size_t N = zisa::pow<Dim - 1>(grid.N_phys) * grid.N_fourier;
    curand_allocate_state<RNG>(&state_, N, seed);
  }
  WhiteNoise(const WhiteNoise &) = default;
  WhiteNoise(WhiteNoise &&) = default;

  ~WhiteNoise() = default;

  WhiteNoise &operator=(const WhiteNoise &) = default;
  WhiteNoise &operator=(WhiteNoise &&) = default;

  void destroy() { curand_free_state<RNG>(state_); }

  __device__ __inline__ void operator()(real_t t, long, complex_t *f1) {
#ifdef __NVCC__
    const size_t id = blockDim.x * blockIdx.x + threadIdx.x;
    state_t local_state = state_[id];
    const auto n = normal2(&local_state);
    state_[id] = local_state;
    f1->x = sigma_ * n.x;
    f1->y = sigma_ * n.y;
#else
    f1->x = 0;
    f1->y = 0;
#endif
  }

  __device__ __inline__ void
  operator()(real_t, long, long, complex_t *f1, complex_t *f2) {
#ifdef __NVCC__
    const size_t idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t id = Ny_ * idx_x + idx_y;
    state_t local_state = state_[id];
    const auto n1 = normal2(&local_state);
    const auto n2 = normal2(&local_state);
    state_[id] = local_state;
    f1->x = sigma_ * n1.x;
    f1->y = sigma_ * n1.y;
    f2->x = sigma_ * n2.x;
    f2->y = sigma_ * n2.y;
#else
    f1->x = 0;
    f1->y = 0;
    f2->x = 0;
    f2->y = 0;
#endif
  }

  __device__ __inline__ void operator()(
      real_t, long, long, long, complex_t *f1, complex_t *f2, complex_t *f3) {
#ifdef __NVCC__
    const size_t idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t idx_z = blockDim.z * blockIdx.z + threadIdx.z;
    const size_t id = Nz_ * (Ny_ * idx_x + idx_y) + idx_z;
    state_t local_state = state_[id];
    const auto n1 = normal2(&local_state);
    const auto n2 = normal2(&local_state);
    const auto n3 = normal2(&local_state);
    state_[id] = local_state;
    f1->x = sigma_ * n1.x;
    f1->y = sigma_ * n1.y;
    f2->x = sigma_ * n2.x;
    f2->y = sigma_ * n2.y;
    f3->x = sigma_ * n3.x;
    f3->y = sigma_ * n3.y;
#else
    f1->x = 0;
    f1->y = 0;
    f2->x = 0;
    f2->y = 0;
    f3->x = 0;
    f3->y = 0;
#endif
  }

private:
  state_t *state_;
  real_t sigma_;
  zisa::int_t Nx_;
  zisa::int_t Ny_;
  zisa::int_t Nz_;

  static __device__ __inline__ float2 normal2(RNG *rng, float) {
    return curand_normal2(rng);
  }

  static __device__ __inline__ double2 normal2(RNG *rng, double) {
    return curand_normal2_double(rng);
  }

  static __device__ __inline__ auto normal2(RNG *rng)
      -> decltype(normal2(rng, real_t{})) {
    return normal2(rng, real_t{});
  }
};

#endif

}

#endif
