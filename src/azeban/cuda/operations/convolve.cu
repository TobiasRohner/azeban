#include <azeban/cuda/operations/convolve_cuda.hpp>


namespace azeban {


__global__ void square_and_scale_kernel(zisa::array_view<real_t, 1> u, real_t scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < u.shape(0)) {
    const real_t ui = u[i];
    u[i] = scale * ui * ui;
  }
}


void square_and_scale_cuda(const zisa::array_view<real_t, 1> &u, real_t scale) {
  const int thread_dims = 1024;
  const int block_dims = zisa::min(zisa::div_up(static_cast<int>(u.shape(0)), thread_dims), 1024);
  square_and_scale_kernel<<<thread_dims, block_dims>>>(u, scale);
}


}
