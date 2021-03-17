#include <azeban/init/double_shear_layer.hpp>
#include <azeban/operations/fft.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

void DoubleShearLayer::initialize(const zisa::array_view<real_t, 3> &u) const {
  const auto init = [&](auto &&u_) {
    const zisa::int_t N = u_.shape(1);
    for (zisa::int_t i = 0; i < N; ++i) {
      for (zisa::int_t j = 0; j < N; ++j) {
        const real_t x = static_cast<real_t>(i) / N;
        const real_t y = static_cast<real_t>(j) / N;
        if (y < 0.5) {
          u_(0, i, j) = std::tanh(2 * zisa::pi * (y - 0.25) / rho_);
        } else {
          u_(0, i, j) = std::tanh(2 * zisa::pi * (0.75 - y) / rho_);
        }
        u_(1, i, j) = delta_ * zisa::sin(2 * zisa::pi * x);
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

void DoubleShearLayer::initialize(
    const zisa::array_view<complex_t, 3> &u_hat) const {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 3>(zisa::shape_t<3>(2, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<2>(u_hat, u);
  initialize(u);
  fft->forward();
}

}
