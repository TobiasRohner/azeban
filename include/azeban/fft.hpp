#ifndef FFT_H_WIQBB
#define FFT_H_WIQBB

#include <azeban/config.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

void fft(const zisa::array_view<complex_t, 3> &u_hat, const zisa::array_const_view<real_t, 3> &u);


}

#endif
