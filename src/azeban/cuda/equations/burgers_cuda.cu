#include <azeban/cuda/equations/burgers_cuda.hpp>
#include <azeban/cuda/equations/burgers_cuda_impl.cuh>



namespace azeban {


#define AZEBAN_INSTANTIATE_BURGERS_CUDA(TYPE)					  \
  template void burgers_cuda<TYPE>(const zisa::array_view<complex_t, 1>&,	  \
				   const zisa::array_const_view<complex_t, 1>&,	  \
				   const TYPE&);

AZEBAN_INSTANTIATE_BURGERS_CUDA(Step1D)
AZEBAN_INSTANTIATE_BURGERS_CUDA(SmoothCutoff1D)

#undef AZEBAN_INSTANTIATE_BURGERS_CUDA


}
