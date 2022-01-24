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
#include <azeban/config.hpp>
#include <azeban/operations/leray.hpp>
#include <azeban/profiler.hpp>
#include <zisa/config.hpp>
#if ZISA_HAS_CUDA
#include <azeban/cuda/operations/leray_cuda.hpp>
#endif

namespace azeban {

void leray(const zisa::array_view<complex_t, 3> &u_hat) {
  AZEBAN_PROFILE_START("leray");
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    const long N_phys = u_hat.shape(1);
    const long N_fourier = N_phys / 2 + 1;
    for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
      long i_ = i;
      if (i_ >= N_fourier) {
        i_ -= N_phys;
      }
      for (zisa::int_t j = 0; j < u_hat.shape(2); ++j) {
        const real_t k1 = 2 * zisa::pi * i_;
        const real_t k2 = 2 * zisa::pi * j;
        const real_t absk2 = k1 * k1 + k2 * k2;
        if (absk2 == 0) {
          u_hat(0, i, j) = 0;
          u_hat(1, i, j) = 0;
        } else {
          const complex_t u1_hat = u_hat(0, i, j);
          const complex_t u2_hat = u_hat(1, i, j);
          u_hat(0, i, j) = (1. - (k1 * k1) / absk2) * u1_hat
                           + (0. - (k1 * k2) / absk2) * u2_hat;
          u_hat(1, i, j) = (0. - (k2 * k1) / absk2) * u1_hat
                           + (1. - (k2 * k2) / absk2) * u2_hat;
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
  AZEBAN_PROFILE_STOP("leray");
}

void leray(const zisa::array_view<complex_t, 4> &u_hat) {
  AZEBAN_PROFILE_START("leray");
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    const long N_phys = u_hat.shape(1);
    const long N_fourier = N_phys / 2 + 1;
    for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
      long i_ = i;
      if (i_ >= N_fourier) {
        i_ -= N_phys;
      }
      for (zisa::int_t j = 0; j < u_hat.shape(2); ++j) {
        long j_ = j;
        if (j_ >= N_fourier) {
          j_ -= N_phys;
        }
        for (zisa::int_t k = 0; k < u_hat.shape(3); ++k) {
          const real_t k1 = 2 * zisa::pi * i_;
          const real_t k2 = 2 * zisa::pi * j_;
          const real_t k3 = 2 * zisa::pi * k;
          const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;
          const complex_t u1_hat = u_hat(0, i, j, k);
          const complex_t u2_hat = u_hat(1, i, j, k);
          const complex_t u3_hat = u_hat(2, i, j, k);
          u_hat(0, i, j, k) = absk2 == 0
                                  ? 0.
                                  : (1. - (k1 * k1) / absk2) * u1_hat
                                        + (0. - (k1 * k2) / absk2) * u2_hat
                                        + (0. - (k1 * k3) / absk2) * u3_hat;
          u_hat(1, i, j, k) = absk2 == 0
                                  ? 0.
                                  : (0. - (k2 * k1) / absk2) * u1_hat
                                        + (1. - (k2 * k2) / absk2) * u2_hat
                                        + (0. - (k2 * k3) / absk2) * u3_hat;
          u_hat(2, i, j, k) = absk2 == 0
                                  ? 0.
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
  AZEBAN_PROFILE_STOP("leray");
}

}
