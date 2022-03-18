#include <azeban/operations/inverse_curl.hpp>
#include <azeban/profiler.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/operations/inverse_curl_cuda.hpp>
#endif

namespace azeban {

void inverse_curl(const zisa::array_const_view<complex_t, 2> &w,
                  const zisa::array_view<complex_t, 3> &u) {
  LOG_ERR_IF(w.memory_location() != u.memory_location(),
             "Memory Locations mismatch");
  LOG_ERR_IF(w.shape(0) != u.shape(1) || w.shape(1) != u.shape(2),
             "Shape mismatch");
  ProfileHost profile("inverse_curl");
  if (w.memory_location() == zisa::device_type::cpu) {
    const long N_phys = w.shape(0);
    const long N_fourier = N_phys / 2 + 1;
    for (zisa::int_t i = 0; i < w.shape(0); ++i) {
      long i_ = i;
      if (i_ >= N_fourier) {
        i_ -= N_phys;
      }
      for (zisa::int_t j = 0; j < w.shape(1); ++j) {
        const real_t k1 = 2 * zisa::pi * i_;
        const real_t k2 = 2 * zisa::pi * j;
        const real_t absk2 = k1 * k1 + k2 * k2;
        if (absk2 == 0) {
          u(0, i, j) = 0;
          u(1, i, j) = 0;
        } else {
          u(0, i, j) = complex_t(0, k2 / absk2) * w(i, j);
          u(1, i, j) = complex_t(0, -k1 / absk2) * w(i, j);
        }
      }
    }
  }
#if ZISA_HAS_CUDA
  else if (w.memory_location() == zisa::device_type::cuda) {
    inverse_curl_cuda(w, u);
  }
#endif
  else {
    LOG_ERR("Unsupported Memory Location");
  }
}

}
