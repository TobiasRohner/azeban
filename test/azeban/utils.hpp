#ifndef TEST_AZEBAN_UTILS_HPP
#define TEST_AZEBAN_UTILS_HPP

#include <azeban/config.hpp>
#include <azeban/operations/copy_padded.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view.hpp>


template<typename T, int D>
zisa::array_view<T, D - 1> component(const zisa::array_view<T, D> &arr, zisa::int_t n) {
  zisa::shape_t<D - 1> slice_shape;
  for (zisa::int_t i = 0 ; i < D - 1 ; ++i) {
    slice_shape[i] = arr.shape(i + 1);
  }
  return zisa::array_view<T, D - 1>(slice_shape, arr.raw() + n * zisa::product(slice_shape), arr.memory_location());
}

template<typename T, int D>
zisa::array_const_view<T, D - 1> component(const zisa::array_const_view<T, D> &arr, zisa::int_t n) {
  zisa::shape_t<D - 1> slice_shape;
  for (zisa::int_t i = 0 ; i < D - 1 ; ++i) {
    slice_shape[i] = arr.shape(i + 1);
  }
  return zisa::array_const_view<T, D - 1>(slice_shape, arr.raw() + n * zisa::product(slice_shape), arr.memory_location());
}

template<typename T, int D>
zisa::array_view<T, D - 1> component(zisa::array<T, D> &arr, zisa::int_t n) {
  zisa::shape_t<D - 1> slice_shape;
  for (zisa::int_t i = 0 ; i < D - 1 ; ++i) {
    slice_shape[i] = arr.shape(i + 1);
  }
  return zisa::array_view<T, D - 1>(slice_shape, arr.raw() + n * zisa::product(slice_shape), arr.device());
}

template<zisa::int_t Dim>
azeban::real_t L2(const zisa::array_const_view<azeban::complex_t, Dim + 1> &u, const zisa::array_const_view<azeban::complex_t, Dim + 1> &u_ref) {
  static_assert(Dim >= 2, "L2 error only supported for dimensions strictly larger than 1");

  LOG_ERR_IF(u.memory_location() != zisa::device_type::cpu, "u must be in host memory");
  LOG_ERR_IF(u_ref.memory_location() != zisa::device_type::cpu, "u_ref must be in host memory");
  LOG_ERR_IF(u.shape(0) != u_ref.shape(0), "Mismatch in number of components");

  const zisa::int_t N = u.shape(1);
  const zisa::int_t N_ref = u_ref.shape(1);

  zisa::array<azeban::complex_t, Dim + 1> u_pad(u_ref.shape());
  for (zisa::int_t d = 0 ; d < u.shape(0) ; ++d) {
    azeban::copy_to_padded(component(u_pad, d), component(u, d), 0);
  }
  for (zisa::int_t i = 0 ; i < u_pad.size() ; ++i) {
    u_pad[i] *= zisa::pow<Dim>(static_cast<azeban::real_t>(N_ref) / N);
  }

  azeban::real_t errL2 = 0;
  if constexpr (Dim == 2) {
    for (zisa::int_t d = 0 ; d < u_ref.shape(0) ; ++d) {
      for (zisa::int_t i = 0 ; i < u_ref.shape(1) ; ++i) {
	errL2 += azeban::abs2(u_pad(d, i, 0) - u_ref(d, i, 0));
	for (zisa::int_t j = 1 ; j < u_ref.shape(2) ; ++j) {
	  errL2 += 2 * azeban::abs2(u_pad(d, i, j) - u_ref(d, i, j));
	}
      }
    }
  } else if constexpr (Dim == 3) {
    for (zisa::int_t d = 0 ; d < u_ref.shape(0) ; ++d) {
      for (zisa::int_t i = 0 ; i < u_ref.shape(1) ; ++i) {
	for (zisa::int_t j = 0 ; j < u_ref.shape(2) ; ++j) {
	  errL2 += azeban::abs2(u_pad(d, i, j, 0) - u_ref(d, i, j, 0));
	  for (zisa::int_t k = 1 ; k < u_ref.shape(3) ; ++k) {
	    errL2 += 2 * azeban::abs2(u_pad(d, i, j, k) - u_ref(d, i, j, k));
	  }
	}
      }
    }
  }
  errL2 = zisa::sqrt(errL2) / zisa::pow<Dim>(N_ref);

  return errL2;
}


#endif
