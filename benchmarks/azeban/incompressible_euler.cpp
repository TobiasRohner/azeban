#include <azeban/benchmark.hpp>

#include <azeban/equations/incompressible_euler.hpp>
#include <azeban/grid_factory.hpp>
#include <cuda_runtime.h>
#include <zisa/cuda/memory/cuda_array.hpp>
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

static void compute_B_2d_params(benchmark::internal::Benchmark *bm) {
  const auto candidates = good_sizes(zisa::int_t(1) << 12);
  for (zisa::int_t N : candidates) {
    bm->Args({N});
  }
}

static void compute_B_3d_params(benchmark::internal::Benchmark *bm) {
  const auto candidates = good_sizes(250);
  for (zisa::int_t N : candidates) {
    bm->Args({N});
  }
}

template <int Dim>
static void bm_compute_B(benchmark::State &state) {
  const zisa::int_t N = state.range(0);
  const auto grid = azeban::make_grid<Dim>(
      nlohmann::json({{"N_phys", N}, {"N_phys_pad", ""}}),
      zisa::device_type::cuda);
  auto h_u = zisa::array<azeban::real_t, Dim + 1>(grid.shape_phys_pad(Dim));
  auto d_u
      = zisa::cuda_array<azeban::real_t, Dim + 1>(grid.shape_phys_pad(Dim));
  auto d_B = zisa::cuda_array<azeban::real_t, Dim + 1>(
      grid.shape_phys_pad((Dim * Dim + Dim) / 2));
  zisa::fill(h_u, azeban::real_t(1));
  zisa::copy(d_u, h_u);

  for (auto _ : state) {
    azeban::incompressible_euler_compute_B_cuda<Dim>(d_B, d_u, grid);
    cudaDeviceSynchronize();
  }
}

BENCHMARK_TEMPLATE(bm_compute_B, 2)->Apply(compute_B_2d_params);
BENCHMARK_TEMPLATE(bm_compute_B, 3)->Apply(compute_B_3d_params);
