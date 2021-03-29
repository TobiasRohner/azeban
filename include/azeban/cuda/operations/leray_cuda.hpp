#ifndef LERAY_CUDA_H_
#define LERAY_CUDA_H_

#include <azeban/config.hpp>
#include <zisa/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

extern void leray_cuda(const zisa::array_view<complex_t, 3> &u_hat);
extern void leray_cuda(const zisa::array_view<complex_t, 4> &u_hat);

}

#endif
