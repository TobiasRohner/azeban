#include <azeban/benchmark.hpp>

#include <azeban/fft.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/cuda/memory/cuda_array.hpp>



static void cufft_1d_sizes(benchmark::internal::Benchmark *bm) {
  for (zisa::int_t i = 128 ; i < zisa::int_t(128)<<10 ; i <<= 1) {
    const zisa::int_t n = zisa::int_t(3./2 * i) & (-2);
    for (zisa::int_t j = n ; j <= 2*i ; j += 2) {
      bm->Args({j});
    }
  }
}


static void bm_cufft_forward_1d(benchmark::State &state) {
  const zisa::int_t n = state.range(0);
  const zisa::shape_t<1> rshape(n);
  const zisa::shape_t<1> cshape(n/2+1);
  auto h_u = zisa::array<azeban::real_t, 1>(rshape);
  auto d_u = zisa::cuda_array<azeban::real_t, 1>(rshape);
  auto d_u_hat = zisa::cuda_array<azeban::complex_t, 1>(cshape);

  for (zisa::int_t i = 0 ; i < n ; ++i) {
    h_u[i] = i < n/2 ? 1 : 0;
  }
  zisa::copy(d_u, h_u);

  const auto fft = azeban::make_fft(zisa::array_view<azeban::complex_t, 1>(d_u_hat),
				    zisa::array_view<azeban::real_t, 1>(d_u));

  for (auto _ : state) {
    fft->forward();
  }
}


BENCHMARK(bm_cufft_forward_1d)->Apply(cufft_1d_sizes);
