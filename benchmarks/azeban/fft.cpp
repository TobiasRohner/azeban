/*
 * This file is part of azeban (https://github.com/TobiasRohner/azeban).
 * Copyright (c) 2021 Tobias Rohner.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include <azeban/benchmark.hpp>

#include <algorithm>
#include <azeban/operations/fft_factory.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <zisa/cuda/memory/cuda_array.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/cuda/operations/cufft_mpi.hpp>
#endif

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
    for (long N : candidates) {
      bm->Args({N, device});
    }
  }
}

static void fft_2d_params(benchmark::internal::Benchmark *bm) {
  const auto candidates = good_sizes(zisa::int_t(1) << 11);
  for (long device : {static_cast<long>(zisa::device_type::cpu),
                      static_cast<long>(zisa::device_type::cuda)}) {
    for (long N : candidates) {
      bm->Args({N, device});
    }
  }
}

static void fft_3d_params(benchmark::internal::Benchmark *bm) {
  const auto candidates
      = good_sizes(220 /* TODO: Change this if GPU memory is larger! */);
  for (long device : {static_cast<long>(zisa::device_type::cpu),
                      static_cast<long>(zisa::device_type::cuda)}) {
    for (long N : candidates) {
      bm->Args({N, device});
    }
  }
}

static void fft_3d_params_mpi(benchmark::internal::Benchmark *bm) {
  const auto candidates = good_sizes(768);
  for (long N : candidates) {
    if (N > 220) {
      bm->Args({N});
    }
  }
}

template <int Dim>
static void bm_fft_forward(benchmark::State &state) {
  const zisa::int_t d = 1;
  const zisa::int_t n = state.range(0);
  const zisa::device_type device
      = static_cast<zisa::device_type>(state.range(1));
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
    cudaDeviceSynchronize();
  }
}

BENCHMARK_TEMPLATE(bm_fft_forward, 1)->Apply(fft_1d_params);
BENCHMARK_TEMPLATE(bm_fft_forward, 2)->Apply(fft_2d_params);
BENCHMARK_TEMPLATE(bm_fft_forward, 3)->Apply(fft_3d_params);

#if AZEBAN_HAS_MPI
static void bm_fft_3d_forward_mpi(benchmark::State &state) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const zisa::int_t d = state.range(0);
  const zisa::int_t n = state.range(1);
  zisa::shape_t<4> rshape{d, n / size + (rank < n % size ? 1 : 0), n, n};
  zisa::shape_t<4> cshape{
      d, n / size + (rank < n % size ? 1 : 0), n / 2 + 1, n};
  auto h_u = zisa::array<azeban::real_t, 4>(rshape);
  auto d_u = zisa::array<azeban::real_t, 4>(rshape, zisa::device_type::cuda);
  auto d_u_hat
      = zisa::array<azeban::complex_t, 4>(cshape, zisa::device_type::cuda);

  for (zisa::int_t i = 0; i < zisa::product(rshape); ++i) {
    h_u[i] = zisa::cos(2.1 * zisa::pi / n * i);
  }
  zisa::copy(d_u, h_u);

  const auto fft = std::make_shared<azeban::CUFFT_MPI<3>>(
      d_u_hat, d_u, MPI_COMM_WORLD, azeban::FFT_FORWARD);

  for (auto _ : state) {
    fft->forward();
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

// BENCHMARK(bm_fft_3d_forward_mpi)->Apply(fft_3d_params_mpi);
#endif
