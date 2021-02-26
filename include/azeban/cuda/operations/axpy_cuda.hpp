#ifndef AXPY_CUDA_H_
#define AXPY_CUDA_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>



namespace azeban {


template<typename Scalar>
void axpy_cuda(const Scalar &a, const zisa::array_const_view<Scalar, 1> &x, const zisa::array_view<Scalar, 1> &y);


#define AZEBAN_INSTANCIATE_AXPY_CUDA(TYPE)					\
  extern template void axpy_cuda<TYPE>(const TYPE&,				\
				       const zisa::array_const_view<TYPE, 1>&,	\
				       const zisa::array_view<TYPE, 1>&);

AZEBAN_INSTANCIATE_AXPY_CUDA(real_t)
AZEBAN_INSTANCIATE_AXPY_CUDA(complex_t)

#undef AZEBAN_INSTANCIATE_AXPY_CUDA


}



#endif
