#include <azeban/init/taylor_vortex.hpp>
#include <azeban/operations/fft.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

void TaylorVortex::initialize(const zisa::array_view<real_t, 3> &u) {
  const auto init = [&](auto &&u_) {
    const zisa::int_t N = u_.shape(1);
    for (zisa::int_t i = 0; i < N; ++i) {
      for (zisa::int_t j = 0; j < N; ++j) {
        const real_t x = 16 * static_cast<real_t>(i) / N - 8;
        const real_t y = 16 * static_cast<real_t>(j) / N - 8;
        u_(0, i, j)
            = (-y * zisa::exp(0.5 * (1 - zisa::pow<2>(x) - zisa::pow<2>(y)))
               + 8)
              / 16;
        u_(1, i, j)
            = x * zisa::exp(0.5 * (1 - zisa::pow<2>(x) - zisa::pow<2>(y))) / 16;
      }
    }
  };
  if (u.memory_location() == zisa::device_type::cpu) {
    init(u);
  } else if (u.memory_location() == zisa::device_type::cuda) {
    auto h_u = zisa::array<real_t, 3>(u.shape(), zisa::device_type::cpu);
    init(h_u);
    zisa::copy(u, h_u);
  } else {
    LOG_ERR("Unknown Memory Location");
  }
}

void TaylorVortex::initialize(const zisa::array_view<complex_t, 3> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 3>(zisa::shape_t<3>(2, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<2>(u_hat, u);
  initialize(u);
  fft->forward();
}

}
