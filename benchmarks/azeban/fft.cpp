#include <azeban/benchmark.hpp>

#include <algorithm>
#include <azeban/operations/fft.hpp>
#include <vector>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>

static zisa::int_t intpow(zisa::int_t b, zisa::int_t e) {
  zisa::int_t result = 1;
  for (;;) {
    if (e & 1) {
      result *= b;
    }
    e >>= 1;
    if (!e) {
      break;
    }
    b *= b;
  }
  return result;
}

static std::vector<zisa::int_t> good_sizes(zisa::int_t Nmax) {
  const auto comp_N
      = [](zisa::int_t p2, zisa::int_t p3, zisa::int_t p5, zisa::int_t p7) {
          return intpow(2, p2) * intpow(3, p3) * intpow(5, p5) * intpow(7, p7);
        };
  std::vector<zisa::int_t> result;
  zisa::int_t p2 = 0;
  zisa::int_t p3 = 0;
  zisa::int_t p5 = 0;
  zisa::int_t p7 = 0;
  zisa::int_t N = comp_N(p2, p3, p5, p7);
  for (;;) {
    result.push_back(N);
    N = comp_N(p2 + 1, p3, p5, p7);
    if (N <= Nmax) {
      ++p2;
      continue;
    }
    N = comp_N(0, p3 + 1, p5, p7);
    if (N <= Nmax) {
      p2 = 0;
      ++p3;
      continue;
    }
    N = comp_N(0, 0, p5 + 1, p7);
    if (N <= Nmax) {
      p2 = 0;
      p3 = 0;
      ++p5;
      continue;
    }
    N = comp_N(0, 0, 0, p7 + 1);
    if (N <= Nmax) {
      p2 = 0;
      p3 = 0;
      p5 = 0;
      ++p7;
      continue;
    }
    std::sort(result.begin(), result.end());
    return result;
  }
}

static void fft_1d_params(benchmark::internal::Benchmark *bm) {
  const auto candidates = good_sizes(zisa::int_t(1) << 14);
  for (long device : {static_cast<long>(zisa::device_type::cpu),
                      static_cast<long>(zisa::device_type::cuda)}) {
    for (zisa::int_t N : candidates) {
      bm->Args({1, N, device});
    }
  }
}

static void fft_2d_params(benchmark::internal::Benchmark *bm) {
  const auto candidates = good_sizes(zisa::int_t(1) << 11);
  for (long device : {static_cast<long>(zisa::device_type::cpu),
                      static_cast<long>(zisa::device_type::cuda)}) {
    for (zisa::int_t d : {2, 3}) {
      for (zisa::int_t N : candidates) {
        bm->Args({d, N, device});
      }
    }
  }
}

static void fft_3d_params(benchmark::internal::Benchmark *bm) {
  const auto candidates = good_sizes(zisa::int_t(1) << 8);
  for (long device : {static_cast<long>(zisa::device_type::cpu),
                      static_cast<long>(zisa::device_type::cuda)}) {
    for (zisa::int_t d : {3, 6}) {
      for (zisa::int_t N : candidates) {
        bm->Args({d, N, device});
      }
    }
  }
}

template <int Dim>
static void bm_fft_forward(benchmark::State &state) {
  const zisa::int_t d = state.range(0);
  const zisa::int_t n = state.range(1);
  const zisa::device_type device
      = static_cast<zisa::device_type>(state.range(2));
  zisa::shape_t<Dim + 1> rshape;
  zisa::shape_t<Dim + 1> cshape;
  rshape[0] = d;
  cshape[0] = d;
  for (int i = 0; i < Dim - 1; ++i) {
    rshape[i + 1] = n;
    cshape[i + 1] = n;
  }
  rshape[Dim] = n;
  cshape[Dim] = n / 2 + 1;
  auto h_u = zisa::array<azeban::real_t, Dim + 1>(rshape);
  auto d_u = zisa::array<azeban::real_t, Dim + 1>(rshape, device);
  auto d_u_hat = zisa::array<azeban::complex_t, Dim + 1>(cshape, device);

  for (zisa::int_t i = 0; i < zisa::product(rshape); ++i) {
    h_u[i] = zisa::cos(2.1 * zisa::pi / n * i);
  }
  zisa::copy(d_u, h_u);

  const auto fft = azeban::make_fft<Dim>(d_u_hat, d_u);

  for (auto _ : state) {
    fft->forward();
  }
}

BENCHMARK_TEMPLATE(bm_fft_forward, 1)->Apply(fft_1d_params);
BENCHMARK_TEMPLATE(bm_fft_forward, 2)->Apply(fft_2d_params);
BENCHMARK_TEMPLATE(bm_fft_forward, 3)->Apply(fft_3d_params);
