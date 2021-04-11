#ifndef INCOMPRESSIBLE_EULER_FUNCTIONS_H_
#define INCOMPRESSIBLE_EULER_FUNCTIONS_H_

#include <azeban/config.hpp>
#include <zisa/config.hpp>

namespace azeban {

ANY_DEVICE_INLINE void incompressible_euler_2d_compute_B(unsigned stride,
                                                         unsigned idx,
                                                         real_t norm,
                                                         real_t u1,
                                                         real_t u2,
                                                         real_t *B) {
  B[0 * stride + idx] = norm * u1 * u1;
  B[1 * stride + idx] = norm * u1 * u2;
  B[2 * stride + idx] = norm * u2 * u2;
}

ANY_DEVICE_INLINE void incompressible_euler_3d_compute_B(unsigned stride,
                                                         unsigned idx,
                                                         real_t norm,
                                                         real_t u1,
                                                         real_t u2,
                                                         real_t u3,
                                                         real_t *B) {
  B[0 * stride + idx] = norm * u1 * u1;
  B[1 * stride + idx] = norm * u2 * u1;
  B[2 * stride + idx] = norm * u2 * u2;
  B[3 * stride + idx] = norm * u3 * u1;
  B[4 * stride + idx] = norm * u3 * u2;
  B[5 * stride + idx] = norm * u3 * u3;
}

ANY_DEVICE_INLINE void incompressible_euler_2d_compute_L(real_t k1,
                                                         real_t k2,
                                                         real_t absk2,
                                                         unsigned stride_B,
                                                         unsigned idx_B,
                                                         const complex_t *B_hat,
                                                         complex_t *L1_hat,
                                                         complex_t *L2_hat) {
  const complex_t B11_hat = B_hat[0 * stride_B + idx_B];
  const complex_t B12_hat = B_hat[1 * stride_B + idx_B];
  const complex_t B22_hat = B_hat[2 * stride_B + idx_B];
  const complex_t b1_hat
      = complex_t(0, k1) * B11_hat + complex_t(0, k2) * B12_hat;
  const complex_t b2_hat
      = complex_t(0, k1) * B12_hat + complex_t(0, k2) * B22_hat;

  *L1_hat
      = (1. - (k1 * k1) / absk2) * b1_hat + (0. - (k1 * k2) / absk2) * b2_hat;
  *L2_hat
      = (0. - (k2 * k1) / absk2) * b1_hat + (1. - (k2 * k2) / absk2) * b2_hat;
}

ANY_DEVICE_INLINE void incompressible_euler_3d_compute_L(real_t k1,
                                                         real_t k2,
                                                         real_t k3,
                                                         real_t absk2,
                                                         unsigned stride_B,
                                                         unsigned idx_B,
                                                         const complex_t *B_hat,
                                                         complex_t *L1_hat,
                                                         complex_t *L2_hat,
                                                         complex_t *L3_hat) {
  const complex_t B11_hat = B_hat[0 * stride_B + idx_B];
  const complex_t B21_hat = B_hat[1 * stride_B + idx_B];
  const complex_t B22_hat = B_hat[2 * stride_B + idx_B];
  const complex_t B31_hat = B_hat[3 * stride_B + idx_B];
  const complex_t B32_hat = B_hat[4 * stride_B + idx_B];
  const complex_t B33_hat = B_hat[5 * stride_B + idx_B];
  const complex_t b1_hat = complex_t(0, k1) * B11_hat
                           + complex_t(0, k2) * B21_hat
                           + complex_t(0, k3) * B31_hat;
  const complex_t b2_hat = complex_t(0, k1) * B21_hat
                           + complex_t(0, k2) * B22_hat
                           + complex_t(0, k3) * B32_hat;
  const complex_t b3_hat = complex_t(0, k1) * B31_hat
                           + complex_t(0, k2) * B32_hat
                           + complex_t(0, k3) * B33_hat;

  *L1_hat = (1. - (k1 * k1) / absk2) * b1_hat
            + (0. - (k1 * k2) / absk2) * b2_hat
            + (0. - (k1 * k3) / absk2) * b3_hat;
  *L2_hat = (0. - (k2 * k1) / absk2) * b1_hat
            + (1. - (k2 * k2) / absk2) * b2_hat
            + (0. - (k2 * k3) / absk2) * b3_hat;
  *L3_hat = (0. - (k3 * k1) / absk2) * b1_hat
            + (0. - (k3 * k2) / absk2) * b2_hat
            + (1. - (k3 * k3) / absk2) * b3_hat;
}

}

#endif
