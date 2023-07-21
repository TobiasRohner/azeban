#ifndef AZEBAN_OPERATIONS_SPECTRUM_HPP_
#define AZEBAN_OPERATIONS_SPECTRUM_HPP_

#include <iostream>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/mpi_types.hpp>
#endif

namespace azeban {

template <typename Op>
static std::vector<real_t>
spectrum_cpu(const Grid<1> &grid,
             const zisa::array_const_view<complex_t, 2> &u_hat,
             long k1_offset = 0) {
  std::vector<real_t> spectrum(grid.N_fourier, 0);
  for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
    long k1 = i + k1_offset;
    if (k1 >= zisa::integer_cast<long>(grid.N_fourier)) {
      k1 -= grid.N_phys;
    }
    const long K = std::abs(k1);
    spectrum[K] += Op::eval(grid, k1, u_hat(0, i));
  }
  return spectrum;
}

template <typename Op>
static std::vector<real_t>
spectrum_cpu(const Grid<2> &grid,
             const zisa::array_const_view<complex_t, 3> &u_hat,
             long k1_offset = 0,
             long k2_offset = 0) {
  std::vector<real_t> spectrum(grid.N_fourier, 0);
  for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
    for (zisa::int_t j = 0; j < u_hat.shape(2); ++j) {
      long k1 = i + k1_offset;
      if (k1 >= zisa::integer_cast<long>(grid.N_fourier)) {
        k1 -= grid.N_phys;
      }
      long k2 = j + k2_offset;
      if (k2 >= zisa::integer_cast<long>(grid.N_fourier)) {
        k2 -= grid.N_phys;
      }
      const long K = std::max(std::abs(k1), std::abs(k2));
      spectrum[K] += Op::eval(grid, k1, k2, u_hat(0, i, j), u_hat(1, i, j));
    }
  }
  return spectrum;
}

template <typename Op>
static std::vector<real_t>
spectrum_cpu(const Grid<3> &grid,
             const zisa::array_const_view<complex_t, 4> &u_hat,
             long k1_offset = 0,
             long k2_offset = 0,
             long k3_offset = 0) {
  std::vector<real_t> spectrum(grid.N_fourier, 0);
  for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
    for (zisa::int_t j = 0; j < u_hat.shape(2); ++j) {
      for (zisa::int_t k = 0; k < u_hat.shape(3); ++k) {
        long k1 = i + k1_offset;
        if (k1 >= zisa::integer_cast<long>(grid.N_fourier)) {
          k1 -= grid.N_phys;
        }
        long k2 = j + k2_offset;
        if (k2 >= zisa::integer_cast<long>(grid.N_fourier)) {
          k2 -= grid.N_phys;
        }
        long k3 = k + k3_offset;
        if (k3 >= zisa::integer_cast<long>(grid.N_fourier)) {
          k3 -= grid.N_phys;
        }
        const long K
            = std::max(std::max(std::abs(k1), std::abs(k2)), std::abs(k3));
        spectrum[K] += Op::eval(grid,
                                k1,
                                k2,
                                k3,
                                u_hat(0, i, j, k),
                                u_hat(1, i, j, k),
                                u_hat(2, i, j, k));
      }
    }
  }
  return spectrum;
}

#if AZEBAN_HAS_MPI
template <typename Op>
static std::vector<real_t>
spectrum_cpu(const Grid<1> &grid,
             const zisa::array_const_view<complex_t, 2> &u_hat,
             MPI_Comm comm) {
  const long k1_offset = grid.i_fourier(0, comm);
  const std::vector<real_t> local_spectrum
      = spectrum_cpu<Op>(grid, u_hat, k1_offset);
  std::vector<real_t> spectrum(local_spectrum.size(), 0);
  MPI_Reduce(local_spectrum.data(),
             spectrum.data(),
             local_spectrum.size(),
             mpi_type<real_t>(),
             MPI_SUM,
             0,
             comm);
  return spectrum;
}

template <typename Op>
static std::vector<real_t>
spectrum_cpu(const Grid<2> &grid,
             const zisa::array_const_view<complex_t, 3> &u_hat,
             MPI_Comm comm) {
  const long k1_offset = grid.i_fourier(0, comm);
  const std::vector<real_t> local_spectrum
      = spectrum_cpu<Op>(grid, u_hat, k1_offset);
  std::vector<real_t> spectrum(local_spectrum.size(), 0);
  MPI_Reduce(local_spectrum.data(),
             spectrum.data(),
             local_spectrum.size(),
             mpi_type<real_t>(),
             MPI_SUM,
             0,
             comm);
  return spectrum;
}

template <typename Op>
static std::vector<real_t>
spectrum_cpu(const Grid<3> &grid,
             const zisa::array_const_view<complex_t, 4> &u_hat,
             MPI_Comm comm) {
  const long k1_offset = grid.i_fourier(0, comm);
  const std::vector<real_t> local_spectrum
      = spectrum_cpu<Op>(grid, u_hat, k1_offset);
  std::vector<real_t> spectrum(local_spectrum.size(), 0);
  MPI_Reduce(local_spectrum.data(),
             spectrum.data(),
             local_spectrum.size(),
             mpi_type<real_t>(),
             MPI_SUM,
             0,
             comm);
  return spectrum;
}
#endif

template <typename Op>
std::vector<real_t>
spectrum(const Grid<1> &grid,
         const zisa::array_const_view<complex_t, 2> &u_hat) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    return spectrum_cpu<Op>(grid, u_hat);
  }
#if ZISA_HAS_CUDA
  else if (u_hat.memory_location() == zisa::device_type::cuda) {
    // TODO: Implement
    LOG_ERR("Not yet implemented");
  }
#endif
  else {
    LOG_ERR("Unsupported memory loaction");
  }
}

template <typename Op>
std::vector<real_t>
spectrum(const Grid<2> &grid,
         const zisa::array_const_view<complex_t, 3> &u_hat) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    return spectrum_cpu<Op>(grid, u_hat);
  }
#if ZISA_HAS_CUDA
  else if (u_hat.memory_location() == zisa::device_type::cuda) {
    // TODO: Implement
    LOG_ERR("Not yet implemented");
  }
#endif
  else {
    LOG_ERR("Unsupported memory loaction");
  }
}

template <typename Op>
std::vector<real_t>
spectrum(const Grid<3> &grid,
         const zisa::array_const_view<complex_t, 4> &u_hat) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    return spectrum_cpu<Op>(grid, u_hat);
  }
#if ZISA_HAS_CUDA
  else if (u_hat.memory_location() == zisa::device_type::cuda) {
    // TODO: Implement
    LOG_ERR("Not yet implemented");
  }
#endif
  else {
    LOG_ERR("Unsupported memory loaction");
  }
}

#if AZEBAN_HAS_MPI
template <typename Op>
std::vector<real_t> spectrum(const Grid<1> &grid,
                             const zisa::array_const_view<complex_t, 2> &u_hat,
                             MPI_Comm comm) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    return spectrum_cpu<Op>(grid, u_hat, comm);
  }
#if ZISA_HAS_CUDA
  else if (u_hat.memory_location() == zisa::device_type::cuda) {
    // TODO: Implement
    LOG_ERR("Not yet implemented");
  }
#endif
  else {
    LOG_ERR("Unsupported memory loaction");
  }
}

template <typename Op>
std::vector<real_t> spectrum(const Grid<2> &grid,
                             const zisa::array_const_view<complex_t, 3> &u_hat,
                             MPI_Comm comm) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    return spectrum_cpu<Op>(grid, u_hat, comm);
  }
#if ZISA_HAS_CUDA
  else if (u_hat.memory_location() == zisa::device_type::cuda) {
    // TODO: Implement
    LOG_ERR("Not yet implemented");
  }
#endif
  else {
    LOG_ERR("Unsupported memory loaction");
  }
}

template <typename Op>
std::vector<real_t> spectrum(const Grid<3> &grid,
                             const zisa::array_const_view<complex_t, 4> &u_hat,
                             MPI_Comm comm) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    return spectrum_cpu<Op>(grid, u_hat, comm);
  }
#if ZISA_HAS_CUDA
  else if (u_hat.memory_location() == zisa::device_type::cuda) {
    // TODO: Implement
    LOG_ERR("Not yet implemented");
  }
#endif
  else {
    LOG_ERR("Unsupported memory loaction");
  }
}
#endif

}

#endif
