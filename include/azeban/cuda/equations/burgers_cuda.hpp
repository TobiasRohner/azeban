#ifndef BURGERS_CUDA_H_
#define BURGERS_CUDS_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>
#include <azeban/equations/spectral_viscosity.hpp>



namespace azeban {


template<typename SpectralViscosity>
void burgers_cuda(const zisa::array_view<complex_t, 1> &u,
		  const zisa::array_const_view<complex_t, 1> &u_squared,
		  const SpectralViscosity &visc);


#define AZEBAN_INSTANTIATE_BURGERS_CUDA(TYPE)						\
  extern template void burgers_cuda<TYPE>(const zisa::array_view<complex_t, 1>&,	\
					  const zisa::array_const_view<complex_t, 1>&,	\
					  const TYPE&);

AZEBAN_INSTANTIATE_BURGERS_CUDA(Step1D)

#undef AZEBAN_INSTANTIATE_BURGERS_CUDA


}



#endif
