#ifndef COPY_PADDED_H_
#define COPY_PADDED_H_

#include <zisa/memory/array_view.hpp>
#ifdef ZISA_HAS_CUDA
#include "cuda/copy_padded_cuda.hpp"
#endif

namespace azeban {

void copy_to_padded(const zisa::array_view<complex_t, 1> &dst,
                    const zisa::array_const_view<complex_t, 1> &src,
                    const complex_t &pad_value);
void copy_to_padded(const zisa::array_view<complex_t, 2> &dst,
                    const zisa::array_const_view<complex_t, 2> &src,
                    const complex_t &pad_value);
void copy_to_padded(const zisa::array_view<complex_t, 3> &dst,
                    const zisa::array_const_view<complex_t, 3> &src,
                    const complex_t &pad_value);

}

#endif
