#ifndef AZEBAN_CUDA_OPERATIONS_STRUCTURE_FUNCTION_CUDA_IMPL_CUH_
#define AZEBAN_CUDA_OPERATIONS_STRUCTURE_FUNCTION_CUDA_IMPL_CUH_

#include <azeban/operations/reduce.hpp>
#include <azeban/operations/structure_function_functionals.hpp>
#include <azeban/utils/math.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

namespace detail {

__device__ __inline__ ssize_t periodic_index(ssize_t i, ssize_t N) {
  if (i < 0) {
    return i + N;
  }
  if (i >= N) {
    return i - N;
  }
  return i;
}

}

template <typename Function>
__global__ void
structure_function_cuda_kernel(const zisa::array_const_view<real_t, 2> u,
                               ssize_t max_h,
                               Function func,
                               const zisa::array_view<real_t, 2> sf,
                               ssize_t i0) {
  const ssize_t i = i0 + blockDim.x * blockIdx.x + threadIdx.x;
  const ssize_t N = u.shape(1);
  if (i >= N) {
    return;
  }
  const real_t vol = 1. / N;
  const real_t ui = u(0, i);
  for (ssize_t di = -max_h + 1; di < max_h; ++di) {
    const ssize_t j = i + di;
    const real_t uj = u(0, detail::periodic_index(j, N));
    const ssize_t h = ::azeban::abs(di);
    sf(h, i) += vol * func(ui, uj, di);
  }
}
template <typename Function>
__global__ void
structure_function_cuda_kernel(const zisa::array_const_view<real_t, 3> u,
                               ssize_t max_h,
                               Function func,
                               const zisa::array_view<real_t, 3> sf,
                               ssize_t i0,
                               ssize_t j0) {
  const ssize_t j_loc = blockDim.x * blockIdx.x + threadIdx.x;
  const ssize_t i_loc = blockDim.y * blockIdx.y + threadIdx.y;
  const ssize_t i = i0 + i_loc;
  const ssize_t j = j0 + j_loc;
  const ssize_t N = u.shape(1);
  const size_t h_stride = sf.size() / sf.shape(0);
  const size_t thread_idx = sf.shape(2) * i_loc + j_loc;
  if (i >= N || j >= N) {
    return;
  }
  const real_t vol = 1. / zisa::pow<2>(N);
  const real_t uij = u(0, i, j);
  const real_t vij = u(1, i, j);
  for (ssize_t di = -max_h + 1; di < max_h; ++di) {
    const ssize_t k = i + di;
    for (ssize_t dj = -max_h + 1; dj < max_h; ++dj) {
      const ssize_t l = j + dj;
      const ssize_t pik = detail::periodic_index(k, N);
      const ssize_t pil = detail::periodic_index(l, N);
      const real_t ukl = u(0, pik, pil);
      const real_t vkl = u(1, pik, pil);
      const ssize_t h = zisa::max(::azeban::abs(di), ::azeban::abs(dj));
      sf[h * h_stride + thread_idx] += vol * func(uij, vij, ukl, vkl, di, dj);
    }
  }
}

template <typename Function>
__global__ void
structure_function_cuda_kernel(const zisa::array_const_view<real_t, 4> u,
                               ssize_t max_h,
                               Function func,
                               const zisa::array_view<real_t, 4> sf,
                               ssize_t i0,
                               ssize_t j0,
                               ssize_t k0) {
  const ssize_t k_loc = blockDim.x * blockIdx.x + threadIdx.x;
  const ssize_t j_loc = blockDim.y * blockIdx.y + threadIdx.y;
  const ssize_t i_loc = blockDim.z * blockIdx.z + threadIdx.z;
  const ssize_t i = i0 + i_loc;
  const ssize_t j = j0 + j_loc;
  const ssize_t k = k0 + k_loc;
  const ssize_t N = u.shape(1);
  const size_t h_stride = sf.size() / sf.shape(0);
  const size_t thread_idx
      = sf.shape(2) * sf.shape(3) * i_loc + sf.shape(3) * j_loc + k_loc;
  if (i >= N || j >= N || k >= N) {
    return;
  }
  const real_t vol = 1. / zisa::pow<3>(N);
  const real_t uijk = u(0, i, j, k);
  const real_t vijk = u(1, i, j, k);
  const real_t wijk = u(2, i, j, k);
  for (ssize_t di = -max_h + 1; di < max_h; ++di) {
    const ssize_t l = i + di;
    for (ssize_t dj = -max_h + 1; dj < max_h; ++dj) {
      const ssize_t m = j + dj;
      for (ssize_t dk = -max_h + 1; dk < max_h; ++dk) {
        const ssize_t n = k + dk;
        const ssize_t pil = detail::periodic_index(l, N);
        const ssize_t pim = detail::periodic_index(m, N);
        const ssize_t pin = detail::periodic_index(n, N);
        const real_t ulmn = u(0, pil, pim, pin);
        const real_t vlmn = u(1, pil, pim, pin);
        const real_t wlmn = u(2, pil, pim, pin);
        const ssize_t h = zisa::max(
            zisa::max(::azeban::abs(di), ::azeban::abs(dj)), ::azeban::abs(dk));
        sf[h * h_stride + thread_idx]
            += vol * func(uijk, vijk, wijk, ulmn, vlmn, wlmn, di, dj, dk);
      }
    }
  }
}

template <typename Function>
std::vector<real_t>
structure_function_cuda(const zisa::array_const_view<real_t, 2> &u,
                        ssize_t max_h,
                        const Function &func) {
  const dim3 thread_dims(1024, 1, 1);
  const dim3 block_dims(
      zisa::div_up(static_cast<int>(u.shape(1)), thread_dims.x), 1, 1);
  zisa::array<real_t, 2> sf(
      zisa::shape_t<2>(max_h, thread_dims.x * block_dims.x),
      zisa::device_type::cuda);
  zisa::fill(sf.raw(), sf.device(), sf.size(), real_t(0));
  structure_function_cuda_kernel<<<block_dims, thread_dims>>>(
      u, max_h, func, sf.view(), 0);
  cudaDeviceSynchronize();
  ZISA_CHECK_CUDA_DEBUG;
  std::vector<real_t> ret(max_h);
  for (ssize_t h = 0; h < max_h; ++h) {
    ret[h] = reduce_sum(zisa::array_view<real_t, 1>(
        zisa::shape_t<1>(sf.shape(1)), &sf(h, 0), zisa::device_type::cuda));
  }
  return ret;
}

template <typename Function>
std::vector<real_t>
structure_function_cuda(const zisa::array_const_view<real_t, 3> &u,
                        ssize_t max_h,
                        const Function &func) {
  const dim3 thread_dims(32, 32, 1);
  const dim3 block_dims(
      zisa::div_up(zisa::min(static_cast<int>(u.shape(1)), 128), thread_dims.x),
      zisa::div_up(zisa::min(static_cast<int>(u.shape(2)), 128), thread_dims.y),
      1);
  const dim3 grid_dims(thread_dims.x * block_dims.x,
                       thread_dims.y * block_dims.y,
                       thread_dims.z * block_dims.z);
  zisa::array<real_t, 3> sf(zisa::shape_t<3>(max_h,
                                             thread_dims.y * block_dims.y,
                                             thread_dims.x * block_dims.x),
                            zisa::device_type::cuda);
  zisa::fill(sf.raw(), sf.device(), sf.size(), real_t(0));
  for (ssize_t i0 = 0; i0 < static_cast<ssize_t>(u.shape(1));
       i0 += grid_dims.y) {
    for (ssize_t j0 = 0; j0 < static_cast<ssize_t>(u.shape(2));
         j0 += grid_dims.x) {
      structure_function_cuda_kernel<<<block_dims, thread_dims>>>(
          u, max_h, func, sf.view(), i0, j0);
      cudaDeviceSynchronize();
      ZISA_CHECK_CUDA_DEBUG;
    }
  }
  std::vector<real_t> ret(max_h);
  for (ssize_t h = 0; h < max_h; ++h) {
    ret[h] = reduce_sum(
        zisa::array_view<real_t, 1>(zisa::shape_t<1>(sf.shape(1) * sf.shape(2)),
                                    &sf(h, 0, 0),
                                    zisa::device_type::cuda));
  }
  return ret;
}

template <typename Function>
std::vector<real_t>
structure_function_cuda(const zisa::array_const_view<real_t, 4> &u,
                        ssize_t max_h,
                        const Function &func) {
  const dim3 thread_dims(32, 4, 4);
  const dim3 block_dims(
      zisa::div_up(zisa::min(static_cast<int>(u.shape(1)), 128), thread_dims.x),
      zisa::div_up(zisa::min(static_cast<int>(u.shape(1)), 128), thread_dims.y),
      zisa::div_up(zisa::min(static_cast<int>(u.shape(3)), 128),
                   thread_dims.z));
  const dim3 grid_dims(thread_dims.x * block_dims.x,
                       thread_dims.y * block_dims.y,
                       thread_dims.z * block_dims.z);
  zisa::array<real_t, 4> sf(zisa::shape_t<4>(max_h,
                                             thread_dims.z * block_dims.z,
                                             thread_dims.y * block_dims.y,
                                             thread_dims.x * block_dims.x),
                            zisa::device_type::cuda);
  zisa::fill(sf.raw(), sf.device(), sf.size(), real_t(0));
  for (ssize_t i0 = 0; i0 < static_cast<ssize_t>(u.shape(1));
       i0 += grid_dims.z) {
    for (ssize_t j0 = 0; j0 < static_cast<ssize_t>(u.shape(2));
         j0 += grid_dims.y) {
      for (ssize_t k0 = 0; k0 < static_cast<ssize_t>(u.shape(3));
           k0 += grid_dims.x) {
        structure_function_cuda_kernel<<<block_dims, thread_dims>>>(
            u, max_h, func, sf.view(), i0, j0, k0);
        cudaDeviceSynchronize();
        ZISA_CHECK_CUDA_DEBUG;
      }
    }
  }
  std::vector<real_t> ret(max_h);
  for (ssize_t h = 0; h < max_h; ++h) {
    ret[h] = reduce_sum(zisa::array_view<real_t, 1>(
        zisa::shape_t<1>(sf.shape(1) * sf.shape(2) * sf.shape(3)),
        &sf(h, 0, 0, 0),
        zisa::device_type::cuda));
  }
  return ret;
}

}

#endif
