#ifndef NORM_CUDA_H_
#define NORM_CUDA_H_

#include <zisa/memory/array_view.hpp>
#include <azeban/config.hpp>


namespace azeban {

template<typename Scalar>
real_t norm_cuda(const zisa::array_const_view<Scalar, 1> &data, real_t p);

#define AZEBAN_INSTANTIATE_REDUCE_CUDA(TYPE)					      \
    extern template real_t norm_cuda<TYPE>(const zisa::array_const_view<TYPE, 1>&,    \
					   real_t p);

AZEBAN_INSTANTIATE_REDUCE_CUDA(real_t)
AZEBAN_INSTANTIATE_REDUCE_CUDA(complex_t)

#undef AZEBAN_INSTANTIATE_REDUCE_CUDA

}


#endif
