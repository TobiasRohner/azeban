#ifndef COPY_PADDED_CUDA_H_
#define COPY_PADDED_CUDA_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

void copy_to_padded_cuda(const zisa::array_view<complex_t, 1> &,
                         const zisa::array_const_view<complex_t, 1> &,
                         const complex_t &);
void copy_to_padded_cuda(const zisa::array_view<complex_t, 2> &,
                         const zisa::array_const_view<complex_t, 2> &,
                         const complex_t &);
void copy_to_padded_cuda(const zisa::array_view<complex_t, 3> &,
                         const zisa::array_const_view<complex_t, 3> &,
                         const complex_t &);

}

#endif
