#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#if AZEBAN_HAS_MPI

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>
#include <mpi.h>


namespace azeban {

void transpose(const zisa::array_view<complex_t, 3> &dst, const zisa::array_const_view<complex_t, 3> &src, MPI_Comm comm);
void transpose(const zisa::array_view<complex_t, 4> &dst, const zisa::array_const_view<complex_t, 4> &src, MPI_Comm comm);

}

#endif

#endif
