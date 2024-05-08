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
#ifndef INCOMPRESSIBLE_EULER_FUNCTIONS_H_
#define INCOMPRESSIBLE_EULER_FUNCTIONS_H_

#include <azeban/config.hpp>
#include <zisa/config.hpp>

namespace azeban {

ANY_DEVICE_INLINE void incompressible_euler_2d_compute_B(
    unsigned long stride, unsigned long idx, real_t u1, real_t u2, real_t *B) {
  B[0 * stride + idx] = u1 * u1;
  B[1 * stride + idx] = u1 * u2;
  B[2 * stride + idx] = u2 * u2;
}

ANY_DEVICE_INLINE void incompressible_euler_3d_compute_B(unsigned long stride,
                                                         unsigned long idx,
                                                         real_t u1,
                                                         real_t u2,
                                                         real_t u3,
                                                         real_t *B) {
  B[0 * stride + idx] = u1 * u1;
  B[1 * stride + idx] = u2 * u1;
  B[2 * stride + idx] = u2 * u2;
  B[3 * stride + idx] = u3 * u1;
  B[4 * stride + idx] = u3 * u2;
  B[5 * stride + idx] = u3 * u3;
}

ANY_DEVICE_INLINE void incompressible_euler_2d_compute_L(real_t k1,
                                                         real_t k2,
                                                         real_t absk2,
                                                         unsigned long stride_B,
                                                         unsigned long idx_B,
                                                         const complex_t *B_hat,
                                                         complex_t force1,
                                                         complex_t force2,
                                                         complex_t *L1_hat,
                                                         complex_t *L2_hat) {
  const complex_t B11_hat = B_hat[0 * stride_B + idx_B];
  const complex_t B12_hat = B_hat[1 * stride_B + idx_B];
  const complex_t B22_hat = B_hat[2 * stride_B + idx_B];
  const complex_t b1_hat
      = complex_t(0, k1) * B11_hat + complex_t(0, k2) * B12_hat - force1;
  const complex_t b2_hat
      = complex_t(0, k1) * B12_hat + complex_t(0, k2) * B22_hat - force2;

  const real_t absk2inv = real_t{1.} / absk2;
  const real_t k11 = k1 * k1 * absk2inv;
  const real_t k12 = k1 * k2 * absk2inv;
  const real_t k22 = k2 * k2 * absk2inv;
  *L1_hat = (1. - k11) * b1_hat + (0. - k12) * b2_hat;
  *L2_hat = (0. - k12) * b1_hat + (1. - k22) * b2_hat;
}

ANY_DEVICE_INLINE void incompressible_euler_3d_compute_L(real_t k1,
                                                         real_t k2,
                                                         real_t k3,
                                                         real_t absk2,
                                                         unsigned long stride_B,
                                                         unsigned long idx_B,
                                                         const complex_t *B_hat,
                                                         complex_t force1,
                                                         complex_t force2,
                                                         complex_t force3,
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
                           + complex_t(0, k3) * B31_hat - force1;
  const complex_t b2_hat = complex_t(0, k1) * B21_hat
                           + complex_t(0, k2) * B22_hat
                           + complex_t(0, k3) * B32_hat - force2;
  const complex_t b3_hat = complex_t(0, k1) * B31_hat
                           + complex_t(0, k2) * B32_hat
                           + complex_t(0, k3) * B33_hat - force3;

  const real_t absk2inv = real_t{1.} / absk2;
  const real_t k11 = k1 * k1 * absk2inv;
  const real_t k12 = k1 * k2 * absk2inv;
  const real_t k13 = k1 * k3 * absk2inv;
  const real_t k22 = k2 * k2 * absk2inv;
  const real_t k23 = k2 * k3 * absk2inv;
  const real_t k33 = k3 * k3 * absk2inv;
  *L1_hat = (1. - k11) * b1_hat + (0. - k12) * b2_hat + (0. - k13) * b3_hat;
  *L2_hat = (0. - k12) * b1_hat + (1. - k22) * b2_hat + (0. - k23) * b3_hat;
  *L3_hat = (0. - k13) * b1_hat + (0. - k23) * b2_hat + (1. - k33) * b3_hat;
}

}

#endif
