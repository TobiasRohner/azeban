#ifndef CONVOLVE_CUDA_H_
#define CONVOLVE_CUDA_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

void scale_and_square_cuda(const zisa::array_view<real_t, 1> &u, real_t scale);

}

#endif
