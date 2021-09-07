#include <azeban/init/double_shear_layer.hpp>
#include <azeban/operations/fft.hpp>
#include <vector>
#include <zisa/memory/array.hpp>

namespace azeban {

void DoubleShearLayer::do_initialize(const zisa::array_view<real_t, 3> &u) {
  const auto init = [&](auto &&u_) {
    const zisa::int_t N = u_.shape(1);
    const real_t rho = rho_.get();
    const real_t delta = delta_.get();
    for (zisa::int_t i = 0; i < N; ++i) {
      const real_t x = static_cast<real_t>(i) / N;
      for (zisa::int_t j = 0; j < N; ++j) {
        const real_t y = static_cast<real_t>(j) / N;
        if (rho == 0) {
          u_(0, i, j) = (zisa::abs(y - 0.5) < 0.25) ? 1 : -1;
        } else {
          if (y < 0.5) {
            u_(0, i, j) = std::tanh(2 * zisa::pi * (y - 0.25) / rho);
          } else {
            u_(0, i, j) = std::tanh(2 * zisa::pi * (0.75 - y) / rho);
          }
        }
        u_(1, i, j) = delta * zisa::sin(2 * zisa::pi * x);
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

void DoubleShearLayer::do_initialize(
    const zisa::array_view<complex_t, 3> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 3>(zisa::shape_t<3>(2, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<2>(u_hat, u);
  initialize(u);
  fft->forward();
}

}
