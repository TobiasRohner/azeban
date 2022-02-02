#include <algorithm>
#include <azeban/operations/fft.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdio.h>
#include <tuple>
#include <vector>
#include <zisa/utils/logging.hpp>

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

template <int Dim, typename ScalarU>
zisa::shape_t<Dim + 1>
FFT<Dim, ScalarU>::shape_u_hat(const zisa::shape_t<Dim + 1> &shape_u) const {
  if (std::is_same_v<ScalarU, real_t>) {
    int last_transform_dim = 0;
    for (int i = 0; i < Dim; ++i) {
      if (transform_dims_[i]) {
        last_transform_dim = i;
      }
    }
    zisa::shape_t<Dim + 1> result = shape_u;
    result[last_transform_dim + 1] = shape_u[last_transform_dim + 1] / 2 + 1;
    return result;
  } else {
    return shape_u;
  }
}

template <int Dim, typename ScalarU>
zisa::shape_t<Dim + 1>
FFT<Dim, ScalarU>::shape_u(const zisa::shape_t<Dim + 1> &shape_u_hat) const {
  if (std::is_same_v<ScalarU, real_t>) {
    int last_transform_dim = 0;
    for (int i = 0; i < Dim; ++i) {
      if (transform_dims_[i]) {
        last_transform_dim = i;
      }
    }
    zisa::shape_t<Dim + 1> result = shape_u_hat;
    result[last_transform_dim + 1]
        = 2 * (shape_u_hat[last_transform_dim + 1] - 1);
    return result;
  } else {
    return shape_u_hat;
  }
}

template <int Dim, typename ScalarU>
void FFT<Dim, ScalarU>::initialize(
    const zisa::array_view<complex_t, Dim + 1> &u_hat,
    const zisa::array_view<scalar_u_t, Dim + 1> &u,
    bool allocate_work_area) {
  LOG_ERR_IF(u_hat.shape(0) != u.shape(0),
             "Number of variables does not match");
  u_hat_ = u_hat;
  u_ = u;
  data_dim_ = u_hat_.shape(0);
  do_initialize(u_hat, u, allocate_work_area);
}

template <int Dim, typename ScalarU>
void FFT<Dim, ScalarU>::set_work_area(void *work_area) {
  do_set_work_area(work_area);
}

template class FFT<1, real_t>;
template class FFT<1, complex_t>;
template class FFT<2, real_t>;
template class FFT<2, complex_t>;
template class FFT<3, real_t>;
template class FFT<3, complex_t>;

}
