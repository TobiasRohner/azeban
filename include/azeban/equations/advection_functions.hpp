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
#ifndef ADVECTION_FUNCTIONS_H_
#define ADVECTION_FUNCTIONS_H_

#include <azeban/config.hpp>
#include <zisa/config.hpp>

namespace azeban {

ANY_DEVICE_INLINE void advection_2d_compute_B(unsigned stride,
                                              unsigned idx,
                                              real_t norm,
                                              real_t rho,
                                              real_t u1,
                                              real_t u2,
                                              real_t *B) {
  B[0 * stride + idx] = norm * rho * u1;
  B[1 * stride + idx] = norm * rho * u2;
}

ANY_DEVICE_INLINE void advection_3d_compute_B(unsigned stride,
                                              unsigned idx,
                                              real_t norm,
                                              real_t rho,
                                              real_t u1,
                                              real_t u2,
                                              real_t u3,
                                              real_t *B) {
  B[0 * stride + idx] = norm * rho * u1;
  B[1 * stride + idx] = norm * rho * u2;
  B[2 * stride + idx] = norm * rho * u3;
}

ANY_DEVICE_INLINE void advection_2d(real_t k1,
                                    real_t k2,
                                    unsigned stride_B,
                                    unsigned idx_B,
                                    const complex_t *B_hat,
                                    complex_t *L_hat) {
  const complex_t rhou1_hat = B_hat[0 * stride_B + idx_B];
  const complex_t rhou2_hat = B_hat[1 * stride_B + idx_B];
  *L_hat = complex_t(0, k1) * rhou1_hat + complex_t(0, k2) * rhou2_hat;
}

ANY_DEVICE_INLINE void advection_3d(real_t k1,
                                    real_t k2,
                                    real_t k3,
                                    unsigned stride_B,
                                    unsigned idx_B,
                                    const complex_t *B_hat,
                                    complex_t *L_hat) {
  const complex_t rhou1_hat = B_hat[0 * stride_B + idx_B];
  const complex_t rhou2_hat = B_hat[1 * stride_B + idx_B];
  const complex_t rhou3_hat = B_hat[2 * stride_B + idx_B];
  *L_hat = complex_t(0, k1) * rhou1_hat + complex_t(0, k2) * rhou2_hat
           + complex_t(0, k3) * rhou3_hat;
}

}

#endif
