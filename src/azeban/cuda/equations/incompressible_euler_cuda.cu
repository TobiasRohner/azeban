#include <azeban/cuda/equations/incompressible_euler_cuda.hpp>
#include <azeban/cuda/equations/incompressible_euler_cuda_impl.cuh>

namespace azeban {

#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_COMPUTE_B_CUDA(DIM)            \
  template void incompressible_euler_compute_B_cuda<DIM>(                      \
      const zisa::array_view<real_t, DIM + 1> &,                               \
      const zisa::array_const_view<real_t, DIM + 1> &,                         \
      zisa::int_t,                                                             \
      zisa::int_t);

AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_COMPUTE_B_CUDA(2)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_COMPUTE_B_CUDA(3)

#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_COMPUTE_B_CUDA

#define AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(TYPE)                     \
  template void incompressible_euler_2d_cuda<TYPE>(                            \
      const zisa::array_const_view<complex_t, 3> &,                            \
      const zisa::array_view<complex_t, 3> &,                                  \
      const TYPE &);                                                           \
  template void incompressible_euler_3d_cuda<TYPE>(                            \
      const zisa::array_const_view<complex_t, 4> &,                            \
      const zisa::array_view<complex_t, 4> &,                                  \
      const TYPE &);

AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(Step1D)
AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA(SmoothCutoff1D)

#undef AZEBAN_INSTANTIATE_INCOMPRESSIBLE_EULER_CUDA

}
