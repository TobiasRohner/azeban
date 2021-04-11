#ifndef FFT_MPI_FACTORY_H_
#define FFT_MPI_FACTORY_H_

#include <azeban/config.hpp>
#include <azeban/cuda/operations/cufft_mpi.hpp>
#include <zisa/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
std::shared_ptr<FFT<Dim>>
make_fft_mpi(const zisa::array_view<complex_t, Dim + 1> &u_hat,
             const zisa::array_view<real_t, Dim + 1> &u,
             MPI_Comm comm,
             int direction = FFT_FORWARD | FFT_BACKWARD) {
  if (u_hat.memory_location() == zisa::device_type::cuda
      && u.memory_location() == zisa::device_type::cuda) {
    return std::make_shared<CUFFT_MPI<Dim>>(u_hat, u, comm, direction);
  } else {
    LOG_ERR("Unsupported memory loactions");
  }
}

}

#endif
