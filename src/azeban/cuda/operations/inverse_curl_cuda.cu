#include <azeban/cuda/operations/inverse_curl_cuda.hpp>

namespace azeban {

__global__ void inverse_curl_cuda_kernel(zisa::array_const_view<complex_t, 2> w,
                                         zisa::array_view<complex_t, 3> u) {
  const unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned long N_phys = w.shape(0);
  const unsigned long N_fourier = N_phys / 2 + 1;

  if (i < w.shape(0) && j < w.shape(1)) {
    long i_ = i;
    if (i_ >= N_fourier) {
      i_ -= N_phys;
    }
    const real_t k1 = 2 * zisa::pi * i_;
    const real_t k2 = 2 * zisa::pi * j;
    const real_t absk2 = k1 * k1 + k2 * k2;
    if (absk2 == 0) {
      u(0, i, j) = 0;
      u(1, i, j) = 0;
    } else {
      const complex_t wij = w(i, j);
      u(0, i, j) = complex_t(0, k2 / absk2) * wij;
      u(1, i, j) = complex_t(0, -k1 / absk2) * wij;
    }
  }
}

void inverse_curl_cuda(const zisa::array_const_view<complex_t, 2> &w,
                       const zisa::array_view<complex_t, 3> &u) {
  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(w.shape(0), zisa::integer_cast<zisa::int_t>(thread_dims.x)),
      zisa::div_up(w.shape(1), zisa::integer_cast<zisa::int_t>(thread_dims.y)),
      1);
  inverse_curl_cuda_kernel<<<block_dims, thread_dims>>>(w, u);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
}

}
