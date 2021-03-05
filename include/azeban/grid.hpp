#ifndef GRID_H_
#define GRID_H_

#include <zisa/config.hpp>


namespace azeban {

template<int Dim>
struct Grid {
  static constexpr int dim_v = Dim;

  zisa::int_t N_phys;
  zisa::int_t N_phys_pad;
  zisa::int_t N_fourier;
  zisa::int_t N_fourier_pad;

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
};

}


#endif
