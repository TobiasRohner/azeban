#include <azeban/fft.hpp>
#include <azeban/init/init_3d_from_2d.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

void Init3DFrom2D::initialize(const zisa::array_view<real_t, 4> &u) const {
  const zisa::int_t N = u.shape(1);
  zisa::array<real_t, 3> u2d(zisa::shape_t<3>(u.shape(0), N, N));
  init2d_->initialize(u2d);
  for (zisa::int_t d = 0; d < u.shape(0); ++d) {
    for (zisa::int_t i = 0; i < N; ++i) {
      for (zisa::int_t j = 0; j < N; ++j) {
        for (zisa::int_t k = 0; k < N; ++k) {
          const zisa::int_t i2d = dim_ > 0 ? i : j;
          const zisa::int_t j2d = dim_ > 1 ? j : k;
          u(d, i, j, k) = u2d(d, i2d, j2d);
        }
      }
    }
  }
}

void Init3DFrom2D::initialize(
    const zisa::array_view<complex_t, 4> &u_hat) const {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 4>(zisa::shape_t<4>(u_hat.shape(0), N, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<3>(u_hat, u);
  initialize(u);
  fft->forward();
}

}
