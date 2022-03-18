#ifndef AZEBAN_CUDA_OPERATIONS_INVERSE_CURL_CUDA_HPP_
#define AZEBAN_CUDA_OPERATIONS_INVERSE_CURL_CUDA_HPP_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

void inverse_curl_cuda(const zisa::array_const_view<complex_t, 2> &w,
                       const zisa::array_view<complex_t, 3> &u);

}

#endif
