#ifndef GRID_H_
#define GRID_H_

#include <azeban/config.hpp>
#include <zisa/config.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

template <int Dim>
struct Grid {
  static constexpr int dim_v = Dim;

  zisa::int_t N_phys;
  zisa::int_t N_phys_pad;
  zisa::int_t N_fourier;
  zisa::int_t N_fourier_pad;

  Grid() = default;

  explicit Grid(zisa::int_t _N_phys) {
    N_phys = _N_phys;
    N_phys_pad = 3. / 2 * N_phys + 1;
    N_fourier = N_phys / 2 + 1;
    N_fourier_pad = N_phys_pad / 2 + 1;
  }

  explicit Grid(zisa::int_t _N_phys, zisa::int_t _N_phys_pad) {
    N_phys = _N_phys;
    N_phys_pad = _N_phys_pad;
    N_fourier = N_phys / 2 + 1;
    N_fourier_pad = N_phys_pad / 2 + 1;
  }

  zisa::shape_t<dim_v + 1> shape_phys(zisa::int_t n_vars) const {
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    for (zisa::int_t i = 1; i < dim_v + 1; ++i) {
      shape[i] = N_phys;
    }
    return shape;
  }

  zisa::shape_t<dim_v + 1> shape_fourier(zisa::int_t n_vars) const {
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    for (zisa::int_t i = 1; i < dim_v; ++i) {
      shape[i] = N_phys;
    }
    shape[dim_v] = N_fourier;
    return shape;
  }

  zisa::shape_t<dim_v + 1> shape_phys_pad(zisa::int_t n_vars) const {
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    for (zisa::int_t i = 1; i < dim_v + 1; ++i) {
      shape[i] = N_phys_pad;
    }
    return shape;
  }

  zisa::shape_t<dim_v + 1> shape_fourier_pad(zisa::int_t n_vars) const {
    zisa::shape_t<dim_v + 1> shape;
    shape[0] = n_vars;
    for (zisa::int_t i = 1; i < dim_v; ++i) {
      shape[i] = N_phys_pad;
    }
    shape[dim_v] = N_fourier_pad;
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
