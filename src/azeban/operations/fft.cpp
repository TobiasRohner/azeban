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
#include <algorithm>
#include <azeban/operations/fft.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdio.h>
#include <tuple>
#include <vector>

namespace azeban {

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

static zisa::int_t next_good_size(zisa::int_t Nmin) {
  const auto comp_N
      = [](zisa::int_t p2, zisa::int_t p3, zisa::int_t p5, zisa::int_t p7) {
          return intpow(2, p2) * intpow(3, p3) * intpow(5, p5) * intpow(7, p7);
        };
  zisa::int_t p2 = 0;
  zisa::int_t p3 = 0;
  zisa::int_t p5 = 0;
  zisa::int_t p7 = 0;
  zisa::int_t N = 1;
  std::vector<zisa::int_t> candidates;
  for (p7 = 0; N < Nmin; ++p7) {
    for (p5 = 0; N < Nmin; ++p5) {
      for (p3 = 0; N < Nmin; ++p3) {
        for (p2 = 0; N < Nmin; ++p2) {
          N = comp_N(p2, p3, p5, p7);
        }
        candidates.push_back(N);
        p2 = 0;
        N = comp_N(p2, p3, p5, p7);
      }
      candidates.push_back(N);
      p3 = 0;
      N = comp_N(p2, p3, p5, p7);
    }
    candidates.push_back(N);
    p5 = 0;
    N = comp_N(p2, p3, p5, p7);
  }
  candidates.push_back(N);
  const zisa::int_t min
      = *std::min_element(candidates.begin(), candidates.end());
  LOG_ERR_IF(min < Nmin, "N must not be smaller than Nmin!");
  return min;
}

zisa::int_t optimal_fft_size(const std::string &benchmark_file,
                             zisa::int_t N,
                             int dim,
                             int n_vars,
                             zisa::device_type device) {
  const zisa::int_t next_bigger = next_good_size(N);
  if (benchmark_file.size() == 0) {
    return next_bigger;
  }

  std::ifstream bm_file(benchmark_file);
  nlohmann::json bms;
  bm_file >> bms;

  std::vector<std::tuple<zisa::int_t, real_t>> candidates;
  for (auto &&bm : bms["benchmarks"]) {
    const std::string name = bm["name"];
    real_t t = bm["real_time"];
    int dim_bm;
    int n_vars_bm;
    int N_bm;
    int device_type_bm;
    sscanf(name.c_str(),
           "bm_fft_forward<%d>/%d/%d/%d",
           &dim_bm,
           &n_vars_bm,
           &N_bm,
           &device_type_bm);
    zisa::device_type device_bm
        = static_cast<zisa::device_type>(device_type_bm);
    if (dim_bm == dim && n_vars == n_vars_bm && device_bm == device
        && zisa::integer_cast<zisa::int_t>(N_bm) >= N) {
      candidates.emplace_back(N_bm, t);
    }
  }

  if (candidates.size() == 0) {
    return next_bigger;
  }

  const auto cmp
      = [](auto &&t1, auto &&t2) { return std::get<1>(t1) < std::get<1>(t2); };
  const auto min_it
      = std::min_element(candidates.begin(), candidates.end(), cmp);
  return std::get<0>(*min_it);
}

}
