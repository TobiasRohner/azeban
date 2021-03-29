#ifndef LERAY_H_
#define LERAY_H_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

void leray(const zisa::array_view<complex_t, 3> &u_hat);
void leray(const zisa::array_view<complex_t, 4> &u_hat);

}

#endif
