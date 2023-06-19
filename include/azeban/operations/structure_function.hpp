#ifndef AZEBAN_OPERATIONS_STRUCTURE_FUNCTION_HPP_
#define AZEBAN_OPERATIONS_STRUCTURE_FUNCTION_HPP_

#include <azeban/config.hpp>
#include <azeban/logging.hpp>
#include <azeban/utils/math.hpp>
#include <vector>
#include <zisa/memory/array_view.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/operations/structure_function_cuda.hpp>
#endif

namespace azeban {

namespace detail {

inline ssize_t periodic_index(ssize_t i, ssize_t N) {
  if (i < 0) {
    return i + N;
  }
  if (i >= N) {
    return i - N;
  }
  return i;
}

}

template <typename Function>
std::vector<real_t>
structure_function_cpu(const zisa::array_const_view<real_t, 2> &u,
                       ssize_t max_h,
                       const Function &func) {
  const ssize_t N = u.shape(1);
  const real_t vol = 1. / N;

  std::vector<real_t> sf(max_h);
#pragma omp parallel
  {
    std::vector<real_t> sf_loc(max_h);
#pragma omp for
    for (ssize_t i = 0; i < N; ++i) {
      const real_t ui = u(0, i);
      for (ssize_t di = -max_h + 1; di < max_h; ++di) {
        const ssize_t j = i + di;
        const real_t uj = u(0, detail::periodic_index(j, N));
        const ssize_t h = ::azeban::abs(di);
        sf[h] += vol * func(ui, uj, di);
      }
    }
#pragma omp critical
    {
      for (ssize_t i = 0; i < max_h; ++i) {
        sf[i] += sf_loc[i];
      }
    }
  }
  return sf;
}

template <typename Function>
std::vector<real_t>
structure_function_cpu(const zisa::array_const_view<real_t, 3> &u,
                       ssize_t max_h,
                       const Function &func) {
  const ssize_t N = u.shape(1);
  const real_t vol = 1. / zisa::pow<2>(N);

  std::vector<real_t> sf(max_h);
#pragma omp parallel
  {
    std::vector<real_t> sf_loc(max_h);
#pragma omp for collapse(2)
    for (ssize_t i = 0; i < N; ++i) {
      for (ssize_t j = 0; j < N; ++j) {
        const real_t uij = u(0, i, j);
        const real_t vij = u(1, i, j);
        for (ssize_t di = -max_h + 1; di < max_h; ++di) {
          for (ssize_t dj = -max_h + 1; dj < max_h; ++dj) {
            const ssize_t k = i + di;
            const ssize_t l = j + dj;
            const real_t ukl = u(
                0, detail::periodic_index(k, N), detail::periodic_index(l, N));
            const real_t vkl = u(
                1, detail::periodic_index(k, N), detail::periodic_index(l, N));
            const ssize_t h = zisa::max(::azeban::abs(di), ::azeban::abs(dj));
            sf_loc[h] += vol * func(uij, vij, ukl, vkl, di, dj);
          }
        }
      }
    }
#pragma omp critical
    {
      for (ssize_t i = 0; i < max_h; ++i) {
        sf[i] += sf_loc[i];
      }
    }
  }
  return sf;
}

template <typename Function>
std::vector<real_t>
structure_function_cpu(const zisa::array_const_view<real_t, 4> &u,
                       ssize_t max_h,
                       const Function &func) {
  const ssize_t N = u.shape(1);
  const real_t vol = 1. / zisa::pow<3>(N);

  std::vector<real_t> sf(max_h);
#pragma omp parallel
  {
    std::vector<real_t> sf_loc(max_h);
#pragma omp for collapse(3)
    for (ssize_t i = 0; i < N; ++i) {
      for (ssize_t j = 0; j < N; ++j) {
        for (ssize_t k = 0; k < N; ++k) {
          const real_t uijk = u(0, i, j, k);
          const real_t vijk = u(1, i, j, k);
          const real_t wijk = u(2, i, j, k);
          for (ssize_t di = -max_h + 1; di < max_h; ++di) {
            for (ssize_t dj = -max_h + 1; dj < max_h; ++dj) {
              for (ssize_t dk = -max_h + 1; dk < max_h; ++dk) {
                const ssize_t l = i + di;
                const ssize_t m = j + dj;
                const ssize_t n = k + dk;
                const real_t ulmn = u(0,
                                      detail::periodic_index(l, N),
                                      detail::periodic_index(m, N),
                                      detail::periodic_index(n, N));
                const real_t vlmn = u(1,
                                      detail::periodic_index(l, N),
                                      detail::periodic_index(m, N),
                                      detail::periodic_index(n, N));
                const real_t wlmn = u(2,
                                      detail::periodic_index(l, N),
                                      detail::periodic_index(m, N),
                                      detail::periodic_index(n, N));
                const ssize_t h
                    = zisa::max(zisa::max(::azeban::abs(di), ::azeban::abs(dj)),
                                ::azeban::abs(dk));
                sf_loc[h]
                    += vol
                       * func(uijk, vijk, wijk, ulmn, vlmn, wlmn, di, dj, dk);
              }
            }
          }
        }
      }
    }
#pragma omp critical
    {
      for (ssize_t i = 0; i < max_h; ++i) {
        sf[i] += sf_loc[i];
      }
    }
  }
  return sf;
}

template <ssize_t Dim, typename Function>
std::vector<real_t>
structure_function(const zisa::array_const_view<real_t, Dim + 1> &u,
                   ssize_t max_h,
                   const Function &func) {
  if (u.memory_location() == zisa::device_type::cpu) {
    return structure_function_cpu(u, max_h, func);
  }
#if ZISA_HAS_CUDA
  else if (u.memory_location() == zisa::device_type::cuda) {
    return structure_function_cuda(u, max_h, func);
  }
#endif
  else {
    AZEBAN_ERR("Unsuported memory location");
  }
}

}

#endif
