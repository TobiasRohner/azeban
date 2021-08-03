#include <azeban/init/discontinuous_double_shear_layer.hpp>
#include <azeban/operations/fft.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

void DiscontinuousDoubleShearLayer::do_initialize(
    const zisa::array_view<real_t, 3> &u) {

  int K = 10;
  real_t gamma = 0.025;
  // clang-format off
  std::vector<real_t> amp{
    -0.51002236, 0.5511518, -0.42899464, 0.11925956, -0.06740886,
    -0.26809261, -0.35521887, 0.73182149, 0.64564053, 0.69387785};

  std::vector<real_t> phase{
     0.9255444 ,  0.01220324, -0.49987564,  0.75983886, -0.39190787,
     0.16493003, -0.34619536, -0.75641841, -0.99380907,  0.75411933
  };
  // clang-format on

  const auto fy = [&](real_t x, real_t y) {
    for (int k = 0; k < K; ++k) {
      y += gamma * amp[k] * zisa::sin(2.0 * zisa::pi * k * (x + phase[k]));
    }

    return y;
  };

  const auto init = [&](auto &u_) {
    const zisa::int_t N = u_.shape(1);
    for (zisa::int_t i = 0; i < N; ++i) {
      for (zisa::int_t j = 0; j < N; ++j) {
        const real_t x = static_cast<real_t>(i) / N;
        const real_t y = fy(x, static_cast<real_t>(j) / N);
        u_(0, i, j) = (zisa::abs(y - 0.5) < 0.25) ? u_(0, i, j) = 1.0
                                                  : u_(0, i, j) = -1.0;
        u_(1, i, j) = 0.0;
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

void DiscontinuousDoubleShearLayer::do_initialize(
    const zisa::array_view<complex_t, 3> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 3>(zisa::shape_t<3>(2, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<2>(u_hat, u);
  initialize(u);
  fft->forward();
}
}
