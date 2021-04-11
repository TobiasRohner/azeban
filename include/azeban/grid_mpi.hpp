#ifndef GRID_MPI_H_
#define GRID_MPI_H_


#include <azeban/config.hpp>
#include <zisa/config.hpp>
#include <zisa/memory/array.hpp>
#include <mpi.h>

namespace azeban {

template <int Dim>
struct Grid_MPI {
  static constexpr int dim_v = Dim;

  zisa::int_t N_phys;
  zisa::int_t N_phys_pad;
  zisa::int_t N_fourier;
  zisa::int_t N_fourier_pad;
  MPI_Comm comm;

  Grid_MPI() = default;

  explicit Grid_MPI(MPI_Comm _comm, zisa::int_t _N_phys) {
    N_phys = _N_phys;
    N_phys_pad = 3 * N_phys / 2;
    N_fourier = N_phys / 2 + 1;
    N_fourier_pad = N_phys_pad / 2 + 1;
    comm = _comm;
  }

  explicit Grid_MPI(MPI_Comm _comm, zisa::int_t _N_phys, zisa::int_t _N_phys_pad) {
    N_phys = _N_phys;
    N_phys_pad = _N_phys_pad;
    N_fourier = N_phys / 2 + 1;
    N_fourier_pad = N_phys_pad / 2 + 1;
    comm = _comm;
  }

  zisa::shape_t<dim_v + 1> shape_phys(zisa::int_t n_vars) const {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    shape[1] = N_phys / size + (rank < N_phys % size);
    for (zisa::int_t i = 2; i < dim_v + 1; ++i) {
      shape[i] = N_phys;
    }
    return shape;
  }

  zisa::shape_t<dim_v + 1> shape_fourier(zisa::int_t n_vars) const {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    if (dim_v == 1) {
      shape[1] = N_fourier;
    }
    else if (dim_v == 2) {
      shape[1] = N_fourier / size + (rank < N_fourier % size);
      shape[2] = N_phys;
    }
    else if (dim_v == 3) {
      shape[1] = N_phys / size + (rank < N_phys % size);
      shape[2] = N_fourier;
      shape[3] = N_phys;
    }
    else {
      LOG_ERR("Unsupported Dimension");
    }
    return shape;
  }

  zisa::shape_t<dim_v + 1> shape_phys_pad(zisa::int_t n_vars) const {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    shape[1] = N_phys_pad / size + (rank < N_phys_pad % size);
    for (zisa::int_t i = 2; i < dim_v + 1; ++i) {
      shape[i] = N_phys_pad;
    }
    return shape;
  }

  zisa::shape_t<dim_v + 1> shape_fourier_pad(zisa::int_t n_vars) const {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    if (dim_v == 1) {
      shape[1] = N_fourier_pad;
    }
    else if (dim_v == 2) {
      shape[1] = N_fourier_pad / size + (rank < N_fourier_pad % size);
      shape[2] = N_phys_pad;
    }
    else if (dim_v == 3) {
      shape[1] = N_phys_pad / size + (rank < N_phys_pad % size);
      shape[2] = N_fourier_pad;
      shape[3] = N_phys_pad;
    }
    else {
      LOG_ERR("Unsupported Dimension");
    }
    return shape;
  }

  zisa::array<real_t, dim_v + 1>
  make_array_phys(zisa::int_t n_vars, zisa::device_type device) const {
    return zisa::array<real_t, dim_v + 1>(shape_phys(n_vars), device);
  }

  zisa::array<complex_t, dim_v + 1>
  make_array_fourier(zisa::int_t n_vars, zisa::device_type device) const {
    return zisa::array<complex_t, dim_v + 1>(shape_fourier(n_vars), device);
  }

  zisa::array<real_t, dim_v + 1>
  make_array_phys_pad(zisa::int_t n_vars, zisa::device_type device) const {
    return zisa::array<real_t, dim_v + 1>(shape_phys_pad(n_vars), device);
  }

  zisa::array<complex_t, dim_v + 1>
  make_array_fourier_pad(zisa::int_t n_vars, zisa::device_type device) const {
    return zisa::array<complex_t, dim_v + 1>(shape_fourier_pad(n_vars), device);
  }
};

}


#endif
