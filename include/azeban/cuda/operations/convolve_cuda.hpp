#ifndef CONVOLVE_CUDA_H_
#define CONVOLVE_CUDA_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>



namespace azeban {


void square_and_scale_cuda(const zisa::array_view<real_t, 1> &u, real_t scale);


}



#endif
