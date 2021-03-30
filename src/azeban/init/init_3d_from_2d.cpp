#include <azeban/init/init_3d_from_2d.hpp>
#include <azeban/operations/fft.hpp>
#include <zisa/memory/array.hpp>

namespace azeban {

void Init3DFrom2D::do_initialize(const zisa::array_view<real_t, 4> &u) {
  const auto init = [&](auto &&u_) {
    const zisa::int_t N = u_.shape(1);
    auto u2d = zisa::array<real_t, 3>(zisa::shape_t<3>(2, N, N),
                                      zisa::device_type::cpu);
    init2d_->initialize(u2d);
    for (zisa::int_t d = 0; d < u_.shape(0); ++d) {
      for (zisa::int_t i = 0; i < N; ++i) {
        for (zisa::int_t j = 0; j < N; ++j) {
          for (zisa::int_t k = 0; k < N; ++k) {
            const zisa::int_t d2d = d < dim_ ? d : d - 1;
            const zisa::int_t i2d = dim_ > 0 ? i : j;
            const zisa::int_t j2d = dim_ > 1 ? j : k;
            if (d == dim_) {
              u_(d, i, j, k) = 0;
            } else {
              u_(d, i, j, k) = u2d(d2d, i2d, j2d);
            }
          }
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
    LOG_ERR("Unknown Memory Location");
  }
}

void Init3DFrom2D::do_initialize(const zisa::array_view<complex_t, 4> &u_hat) {
  const zisa::int_t N = u_hat.shape(1);
  auto u = zisa::array<real_t, 4>(zisa::shape_t<4>(u_hat.shape(0), N, N, N),
                                  u_hat.memory_location());
  auto fft = make_fft<3>(u_hat, u);
  initialize(u);
  fft->forward();
}

}
