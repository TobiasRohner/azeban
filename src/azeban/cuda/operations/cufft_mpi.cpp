#include <azeban/cuda/operations/cufft_mpi.hpp>

namespace azeban {

CUFFT_MPI<2>::CUFFT_MPI(const zisa::array_view<complex_t, 3> &u_hat,
                        const zisa::array_view<real_t, 3> &u,
                        int direction)
    : super(u_hat, u, direction) {
  // TODO
}

CUFFT_MPI<2>::~CUFFT_MPI() {
  // TODO
}

void CUFFT_MPI<2>::forward() {
  // TODO
}

void CUFFT_MPI<2>::backward() {
  // TODO
}

}
