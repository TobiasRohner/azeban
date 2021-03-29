#include <azeban/init/discontinuous_vortex_patch.hpp>
#include <azeban/operations/fft.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

void DiscontinuousVortexPatch::do_initialize(
    const zisa::array_view<real_t, 3> &u) const {
  const auto init = [&](auto &&u_) {
    const zisa::int_t N = u_.shape(1);
    for (zisa::int_t i = 0; i < N; ++i) {
      for (zisa::int_t j = 0; j < N; ++j) {
        const real_t x = static_cast<real_t>(i) / N;
        const real_t y = static_cast<real_t>(j) / N;
        const real_t r
            = zisa::sqrt(zisa::pow<2>(x - 0.5) + zisa::pow<2>(y - 0.5));
        u_(0, i, j) = r < 0.25 ? -0.5 * (y - 0.5) : 0;
        u_(1, i, j) = r < 0.25 ? 0.5 * (x - 0.5) : 0;
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

void DiscontinuousVortexPatch::do_initialize(
    const zisa::array_view<complex_t, 3> &u_hat) const {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 3>(zisa::shape_t<3>(2, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<2>(u_hat, u);
  initialize(u);
  fft->forward();
}

}
