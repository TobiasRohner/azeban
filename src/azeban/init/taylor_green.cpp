#include <azeban/init/taylor_green.hpp>
#include <azeban/operations/fft.hpp>
#include <zisa/math/basic_functions.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

void TaylorGreen<2>::do_initialize(const zisa::array_view<real_t, 3> &u) {
  const auto init = [&](auto &&u_) {
    const real_t A = 1;
    const real_t B = -1;
    const zisa::int_t N = u_.shape(1);
    for (zisa::int_t i = 0; i < N; ++i) {
      for (zisa::int_t j = 0; j < N; ++j) {
        const real_t x = 2 * zisa::pi / N * i;
        const real_t y = 2 * zisa::pi / N * j;
        u_(0, i, j) = A * zisa::cos(x) * zisa::sin(y);
        u_(1, i, j) = B * zisa::sin(x) * zisa::cos(y);
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
    LOG_ERR("Unsupported Memory Location");
  }
}

void TaylorGreen<2>::do_initialize(
    const zisa::array_view<complex_t, 3> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 3>(zisa::shape_t<3>(2, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<2>(u_hat, u);
  do_initialize(u);
  fft->forward();
}

void TaylorGreen<3>::do_initialize(const zisa::array_view<real_t, 4> &u) {
  const auto init = [&](auto &&u_) {
    const real_t A = 1;
    const real_t B = -1;
    const real_t C = 0;
    const zisa::int_t N = u_.shape(1);
    for (zisa::int_t i = 0; i < N; ++i) {
      for (zisa::int_t j = 0; j < N; ++j) {
        for (zisa::int_t k = 0; k < N; ++k) {
          const real_t x = 2 * zisa::pi / N * i;
          const real_t y = 2 * zisa::pi / N * j;
          const real_t z = 2 * zisa::pi / N * k;
          u_(0, i, j, k) = A * zisa::cos(x) * zisa::sin(y) * zisa::sin(z);
          u_(1, i, j, k) = B * zisa::sin(x) * zisa::cos(y) * zisa::sin(z);
          u_(2, i, j, k) = C * zisa::sin(x) * zisa::sin(y) * zisa::cos(z);
        }
      }
    }
  };
  if (u.memory_location() == zisa::device_type::cpu) {
    init(u);
  } else if (u.memory_location() == zisa::device_type::cuda) {
    auto h_u = zisa::array<real_t, 4>(u.shape(), zisa::device_type::cpu);
    init(h_u);
    zisa::copy(u, h_u);
  } else {
    LOG_ERR("Unsupported Memory Location");
  }
}

void TaylorGreen<3>::do_initialize(
    const zisa::array_view<complex_t, 4> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 4>(zisa::shape_t<4>(3, N, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<3>(u_hat, u);
  do_initialize(u);
  fft->forward();
}

}
