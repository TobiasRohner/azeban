#include <azeban/cuda/operations/convolve_cuda.hpp>

namespace azeban {

__global__ void scale_and_square_kernel(zisa::array_view<real_t, 1> u,
                                        real_t scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < u.shape(0)) {
    const real_t ui_scaled = scale * u[i];
    u[i] = ui_scaled * ui_scaled;
  }
}

void scale_and_square_cuda(const zisa::array_view<real_t, 1> &u, real_t scale) {
  const int thread_dims = 1024;
  const int block_dims = zisa::min(
      zisa::div_up(static_cast<int>(u.shape(0)), thread_dims), 1024);
  scale_and_square_kernel<<<block_dims, thread_dims>>>(u, scale);
}

}
