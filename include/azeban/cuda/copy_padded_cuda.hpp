#ifndef COPY_PADDED_CUDA_H_
#define COPY_PADDED_CUDA_H_

#include <zisa/memory/array_view.hpp>


namespace azeban {


template<typename T>
void copy_to_padded_cuda(const zisa::array_view<T, 1>&,
			 const zisa::array_const_view<T, 1>&,
			 const T&);
template<typename T>
void copy_to_padded_cuda(const zisa::array_view<T, 2>&,
			 const zisa::array_const_view<T, 2>&,
			 const T&);
template<typename T>
void copy_to_padded_cuda(const zisa::array_view<T, 3>&,
			 const zisa::array_const_view<T, 3>&,
			 const T&);

template<typename T>
void copy_from_padded_cuda(const zisa::array_view<T, 1>&,
			   const zisa::array_const_view<T, 1>&);
template<typename T>
void copy_from_padded_cuda(const zisa::array_view<T, 2>&,
			   const zisa::array_const_view<T, 2>&);
template<typename T>
void copy_from_padded_cuda(const zisa::array_view<T, 3>&,
			   const zisa::array_const_view<T, 3>&);


}



#endif
