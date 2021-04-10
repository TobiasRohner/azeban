#ifndef ADVECTION_FUNCTIONS_H_
#define ADVECTION_FUNCTIONS_H_

#include <azeban/config.hpp>
#include <zisa/config.hpp>


namespace azeban {

ANY_DEVICE_INLINE void advection_2d_compute_B(unsigned stride, unsigned idx, real_t norm, real_t rho, real_t u1, real_t u2, real_t *B) {
  B[0 * stride + idx] = norm * rho * u1;
  B[1 * stride + idx] = norm * rho * u2;
}

ANY_DEVICE_INLINE void advection_3d_compute_B(unsigned stride, unsigned idx, real_t norm, real_t rho, real_t u1, real_t u2, real_t u3, real_t *B) {
  B[0 * stride + idx] = norm * rho * u1;
  B[1 * stride + idx] = norm * rho * u2;
  B[2 * stride + idx] = norm * rho * u3;
}

ANY_DEVICE_INLINE void advection_2d(real_t k1, real_t k2, unsigned stride_B, unsigned idx_B, const complex_t *B_hat, complex_t *L_hat) {
  const complex_t rhou1_hat = B_hat[0 * stride_B + idx_B];
  const complex_t rhou2_hat = B_hat[1 * stride_B + idx_B];
  *L_hat = complex_t(0, k1) * rhou1_hat + complex_t(0, k2) * rhou2_hat;
}

ANY_DEVICE_INLINE void advection_3d(real_t k1, real_t k2, real_t k3, unsigned stride_B, unsigned idx_B, const complex_t *B_hat, complex_t *L_hat) {
  const complex_t rhou1_hat = B_hat[0 * stride_B + idx_B];
  const complex_t rhou2_hat = B_hat[1 * stride_B + idx_B];
  const complex_t rhou3_hat = B_hat[2 * stride_B + idx_B];
  *L_hat = complex_t(0, k1) * rhou1_hat
	    + complex_t(0, k2) * rhou2_hat
	    + complex_t(0, k3) * rhou3_hat;
}

}


#endif
