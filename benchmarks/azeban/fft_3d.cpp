#include <azeban/benchmark.hpp>

static void bm_fft3d(benchmark::State &state) {

  auto run_fft = []() { return true; };

  for (auto _ : state) {
    benchmark::DoNotOptimize(run_fft());
  }
}

BENCHMARK(bm_fft3d);