#include <azeban/operations/energy_spectrum.hpp>
#include <iostream>

namespace azeban {

static std::vector<real_t>
energy_spectrum_cpu(const Grid<1> &grid,
                    const zisa::array_const_view<complex_t, 2> &u_hat) {
  std::vector<real_t> spectrum(grid.N_fourier, 0);
  for (zisa::int_t d = 0; d < 2; ++d) {
    for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
      long k1 = i;
      if (k1 >= zisa::integer_cast<long>(grid.N_fourier)) {
        k1 -= grid.N_phys;
      }
      const long K = std::abs(k1);
      spectrum[K] += 0.5 * abs2(u_hat(d, i)) / zisa::pow<1>(grid.N_phys);
    }
  }
  return spectrum;
}

static std::vector<real_t>
energy_spectrum_cpu(const Grid<2> &grid,
                    const zisa::array_const_view<complex_t, 3> &u_hat) {
  std::vector<real_t> spectrum(grid.N_fourier, 0);
  for (zisa::int_t d = 0; d < 2; ++d) {
    for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
      for (zisa::int_t j = 0; j < u_hat.shape(2); ++j) {
        long k1 = i;
        if (k1 >= zisa::integer_cast<long>(grid.N_fourier)) {
          k1 -= grid.N_phys;
        }
        long k2 = j;
        if (k2 >= zisa::integer_cast<long>(grid.N_fourier)) {
          k2 -= grid.N_phys;
        }
        const long K = std::max(std::abs(k1), std::abs(k2));
        spectrum[K] += 0.5 * abs2(u_hat(d, i, j)) / zisa::pow<2>(grid.N_phys);
      }
    }
  }
  return spectrum;
}

static std::vector<real_t>
energy_spectrum_cpu(const Grid<3> &grid,
                    const zisa::array_const_view<complex_t, 4> &u_hat) {
  std::vector<real_t> spectrum(grid.N_fourier, 0);
  for (zisa::int_t d = 0; d < 3; ++d) {
    for (zisa::int_t i = 0; i < u_hat.shape(1); ++i) {
      for (zisa::int_t j = 0; j < u_hat.shape(2); ++j) {
        for (zisa::int_t k = 0; k < u_hat.shape(3); ++k) {
          long k1 = i;
          if (k1 >= zisa::integer_cast<long>(grid.N_fourier)) {
            k1 -= grid.N_phys;
          }
          long k2 = j;
          if (k2 >= zisa::integer_cast<long>(grid.N_fourier)) {
            k2 -= grid.N_phys;
          }
          long k3 = k;
          if (k3 >= zisa::integer_cast<long>(grid.N_fourier)) {
            k3 -= grid.N_phys;
          }
          const long K
              = std::max(std::max(std::abs(k1), std::abs(k2)), std::abs(k3));
          spectrum[K]
              += 0.5 * abs2(u_hat(d, i, j, k)) / zisa::pow<3>(grid.N_phys);
        }
      }
    }
  }
  return spectrum;
}

std::vector<real_t>
energy_spectrum(const Grid<1> &grid,
                const zisa::array_const_view<complex_t, 2> &u_hat) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    return energy_spectrum_cpu(grid, u_hat);
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

std::vector<real_t>
energy_spectrum(const Grid<2> &grid,
                const zisa::array_const_view<complex_t, 3> &u_hat) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    return energy_spectrum_cpu(grid, u_hat);
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

std::vector<real_t>
energy_spectrum(const Grid<3> &grid,
                const zisa::array_const_view<complex_t, 4> &u_hat) {
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    return energy_spectrum_cpu(grid, u_hat);
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

}
