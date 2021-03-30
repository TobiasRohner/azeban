#include <azeban/init/brownian_motion.hpp>

namespace azeban {

void BrownianMotion<1>::do_initialize(const zisa::array_view<real_t, 2> &u) {
  const auto init = [&](auto &&u_, real_t H) {
    const zisa::int_t N = u_.shape(1);
    zisa::shape_t<1> shape_u;
    shape_u[0] = N;
    for (zisa::int_t d = 0; d < u_.shape(0); ++d) {
      zisa::array_view<real_t, 1> view(shape_u,
                                       u_.raw() + d * zisa::product(shape_u),
                                       zisa::device_type::cpu);
      generate_step(view, H, 0, N);
    }
  };
  const real_t H = hurst_.get();
  if (u.memory_location() == zisa::device_type::cpu) {
    init(u, H);
  } else if (u.memory_location() == zisa::device_type::cuda) {
    auto h_u = zisa::array<real_t, 2>(u.shape(), zisa::device_type::cpu);
    init(h_u, H);
    zisa::copy(u, h_u);
  } else {
    LOG_ERR("Unknown Memory Location");
  }
}

void BrownianMotion<1>::do_initialize(
    const zisa::array_view<complex_t, 2> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u
      = zisa::array<real_t, 2>(zisa::shape_t<2>(1, N), u_hat.memory_location());
  auto fft = make_fft<1>(u_hat, u);
  initialize(u);
  fft->forward();
}

void BrownianMotion<1>::generate_step(const zisa::array_view<real_t, 1> &u,
                                      real_t H,
                                      zisa::int_t i0,
                                      zisa::int_t i1) {
  if (i1 - i0 == 1) {
    return;
  }
  const zisa::int_t N = u.shape(0);
  const zisa::int_t im = i0 + (i1 - i0) / 2;
  const real_t ui0 = u(i0);
  const real_t ui1 = i1 == u.shape(0) ? 0 : u(i1);
  const real_t X = normal_.get();
  const real_t sigma = zisa::sqrt((i1 - i0) * (1. - zisa::pow(2., 2 * H - 2))
                                  / zisa::pow(N, 2 * H));
  u(im) = 0.5 * (ui0 + ui1) + sigma * X;
  generate_step(u, H, i0, im);
  generate_step(u, H, im, i1);
}

void BrownianMotion<2>::do_initialize(const zisa::array_view<real_t, 3> &u) {
  const auto init = [&](auto &&u_, real_t H) {
    const zisa::int_t N = u_.shape(1);
    zisa::shape_t<2> shape_u;
    shape_u[0] = N;
    shape_u[1] = N;
    for (zisa::int_t d = 0; d < u_.shape(0); ++d) {
      zisa::array_view<real_t, 2> view(shape_u,
                                       u_.raw() + d * zisa::product(shape_u),
                                       zisa::device_type::cpu);
      generate_step(view, H, 0, N, 0, N);
    }
  };
  const real_t H = hurst_.get();
  if (u.memory_location() == zisa::device_type::cpu) {
    init(u, H);
  } else if (u.memory_location() == zisa::device_type::cuda) {
    auto h_u = zisa::array<real_t, 3>(u.shape(), zisa::device_type::cpu);
    init(h_u, H);
    zisa::copy(u, h_u);
  } else {
    LOG_ERR("Unknown Memory Location");
  }
}

void BrownianMotion<2>::do_initialize(
    const zisa::array_view<complex_t, 3> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 3>(zisa::shape_t<3>(2, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<2>(u_hat, u);
  initialize(u);
  fft->forward();
}

void BrownianMotion<2>::generate_step(const zisa::array_view<real_t, 2> &u,
                                      real_t H,
                                      zisa::int_t i0,
                                      zisa::int_t i1,
                                      zisa::int_t j0,
                                      zisa::int_t j1) {
  LOG_ERR("Not yet implemented");
}

void BrownianMotion<3>::do_initialize(const zisa::array_view<real_t, 4> &u) {
  const auto init = [&](auto &&u_, real_t H) {
    const zisa::int_t N = u_.shape(1);
    zisa::shape_t<3> shape_u;
    shape_u[0] = N;
    shape_u[1] = N;
    shape_u[2] = N;
    for (zisa::int_t d = 0; d < u_.shape(0); ++d) {
      zisa::array_view<real_t, 3> view(shape_u,
                                       u_.raw() + d * zisa::product(shape_u),
                                       zisa::device_type::cpu);
      generate_step(view, H, 0, N, 0, N, 0, N);
    }
  };
  const real_t H = hurst_.get();
  if (u.memory_location() == zisa::device_type::cpu) {
    init(u, H);
  } else if (u.memory_location() == zisa::device_type::cuda) {
    auto h_u = zisa::array<real_t, 4>(u.shape(), zisa::device_type::cpu);
    init(h_u, H);
    zisa::copy(u, h_u);
  } else {
    LOG_ERR("Unknown Memory Location");
  }
}

void BrownianMotion<3>::do_initialize(
    const zisa::array_view<complex_t, 4> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 4>(zisa::shape_t<4>(3, N, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<3>(u_hat, u);
  initialize(u);
  fft->forward();
}

void BrownianMotion<3>::generate_step(const zisa::array_view<real_t, 3> &u,
                                      real_t H,
                                      zisa::int_t i0,
                                      zisa::int_t i1,
                                      zisa::int_t j0,
                                      zisa::int_t j1,
                                      zisa::int_t k0,
                                      zisa::int_t k1) {
  LOG_ERR("Not yet implemented");
}

}
