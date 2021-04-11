#include <azeban/config.hpp>
#include <azeban/operations/leray.hpp>
#include <zisa/config.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/operations/leray_cuda.hpp>
#endif

namespace azeban {

void leray(const zisa::array_view<complex_t, 3> &u_hat) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
      long i_ = i;
      if (i_ >= zisa::integer_cast<long>(u_hat.shape(1) / 2 + 1)) {
        i_ -= u_hat.shape(1);
      }
      for (zisa::int_t j = 0; j < u_hat.shape(2); ++j) {
        long j_ = j;
        if (j_ >= zisa::integer_cast<long>(u_hat.shape(2) / 2 + 1)) {
          j_ -= u_hat.shape(2);
        }
        const real_t k1 = 2 * zisa::pi * i_;
        const real_t k2 = 2 * zisa::pi * j_;
        const real_t absk2 = k1 * k1 + k2 * k2;
        const complex_t u1_hat = u_hat(0, i, j);
        const complex_t u2_hat = u_hat(1, i, j);
        u_hat(0, i, j) = absk2 == 0 ? u1_hat
                                    : (1. - (k1 * k1) / absk2) * u1_hat
                                          + (0. - (k1 * k2) / absk2) * u2_hat;
        u_hat(1, i, j) = absk2 == 0 ? u2_hat
                                    : (0. - (k2 * k1) / absk2) * u1_hat
                                          + (1. - (k2 * k2) / absk2) * u2_hat;
      }
    }
  }
#if ZISA_HAS_CUDA
  else if (u_hat.memory_location() == zisa::device_type::cuda) {
    leray_cuda(u_hat);
  }
#endif
  else {
    LOG_ERR("Unsupported Memory Location");
  }
}

void leray(const zisa::array_view<complex_t, 4> &u_hat) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
      long i_ = i;
      if (i_ >= zisa::integer_cast<long>(u_hat.shape(1) / 2 + 1)) {
        i_ -= u_hat.shape(1);
      }
      for (zisa::int_t j = 0; j < u_hat.shape(2); ++j) {
        long j_ = j;
        if (j_ >= zisa::integer_cast<long>(u_hat.shape(2) / 2 + 1)) {
          j_ -= u_hat.shape(2);
        }
        for (zisa::int_t k = 0; k < u_hat.shape(3); ++k) {
          long k_ = k;
          if (k_ >= zisa::integer_cast<long>(u_hat.shape(3) / 2 + 1)) {
            k_ -= u_hat.shape(3);
          }
          const real_t k1 = 2 * zisa::pi * i_;
          const real_t k2 = 2 * zisa::pi * j_;
          const real_t k3 = 2 * zisa::pi * k_;
          const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;
          const complex_t u1_hat = u_hat(0, i, j, k);
          const complex_t u2_hat = u_hat(1, i, j, k);
          const complex_t u3_hat = u_hat(2, i, j, k);
          u_hat(0, i, j, k) = absk2 == 0
                                  ? u1_hat
                                  : (1. - (k1 * k1) / absk2) * u1_hat
                                        + (0. - (k1 * k2) / absk2) * u2_hat
                                        + (0. - (k1 * k3) / absk2) * u3_hat;
          u_hat(1, i, j, k) = absk2 == 0
                                  ? u2_hat
                                  : (0. - (k2 * k1) / absk2) * u1_hat
                                        + (1. - (k2 * k2) / absk2) * u2_hat
                                        + (0. - (k2 * k3) / absk2) * u3_hat;
          u_hat(2, i, j, k) = absk2 == 0
                                  ? u3_hat
                                  : (0. - (k3 * k1) / absk2) * u1_hat
                                        + (0. - (k3 * k2) / absk2) * u2_hat
                                        + (1. - (k3 * k3) / absk2) * u3_hat;
        }
      }
    }
  }
#if ZISA_HAS_CUDA
  else if (u_hat.memory_location() == zisa::device_type::cuda) {
    leray_cuda(u_hat);
  }
#endif
  else {
    LOG_ERR("Unsupported Memory Location");
  }
}

}
