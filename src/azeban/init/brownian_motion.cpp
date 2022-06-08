/*
 * This file is part of azeban (https://github.com/TobiasRohner/azeban).
 * Copyright (c) 2021 Tobias Rohner.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include <azeban/init/brownian_motion.hpp>
#include <azeban/operations/scale.hpp>

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
      view(0) = 0;
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
  scale(static_cast<real_t>(1. / N), u.view());
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
  const real_t sigma
      = zisa::sqrt(zisa::pow(static_cast<real_t>(i1 - i0) / N, 2 * H)
                   * (1. - zisa::pow(static_cast<real_t>(2.), 2 * H - 2)));
  u(im) = 0.5 * (ui0 + ui1) + sigma * X;
  generate_step(u, H, i0, im);
  generate_step(u, H, im, i1);
}

void BrownianMotion<2>::do_initialize(const zisa::array_view<real_t, 3> &u) {
  const zisa::int_t N = u.shape(1);
  auto u_hat = zisa::array<complex_t, 3>(zisa::shape_t<3>(2, N, N / 2 + 1),
                                         u.memory_location());
  auto fft = make_fft<2>(u_hat, u);
  initialize(u_hat);
  fft->backward();
  scale(static_cast<real_t>(1. / N), u);
  scale(static_cast<real_t>(1. / N), u);
}

void BrownianMotion<2>::do_initialize(
    const zisa::array_view<complex_t, 3> &u_hat) {
  const auto init = [&](auto &&u_hat_, real_t H) {
    const zisa::int_t N = u_hat_.shape(1);
    const zisa::int_t N_fourier = N / 2 + 1;
    for (int d = 0; d < 2; ++d) {
      u_hat_(d, 0, 0) = 0;
      for (zisa::int_t k1 = 0; k1 < N_fourier; ++k1) {
        for (zisa::int_t k2 = 0; k2 < N_fourier; ++k2) {
          if (k1 == 0 && k2 == 0) {
            continue;
          }
          const real_t cc = uniform_.get();
          const real_t cs = uniform_.get();
          const real_t sc = uniform_.get();
          const real_t ss = uniform_.get();
          const real_t fac
              = static_cast<real_t>(N * N)
                / zisa::pow(static_cast<real_t>(4 * zisa::pi * zisa::pi
                                                * (k1 * k1 + k2 * k2)),
                            (H + 1) / 2);
          u_hat_(d, k1, k2) = fac * complex_t(cc - ss, cs + sc);
          if (k1 > 0) {
            u_hat_(d, N - k1, k2) = fac * complex_t(cc + ss, cs - sc);
          }
        }
      }
    }
  };
  const real_t H = hurst_.get();
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    init(u_hat, H);
  } else if (u_hat.memory_location() == zisa::device_type::cuda) {
    auto h_u_hat
        = zisa::array<complex_t, 3>(u_hat.shape(), zisa::device_type::cpu);
    init(h_u_hat, H);
    zisa::copy(u_hat, h_u_hat);
  } else {
    LOG_ERR("Unknown Memory Location");
  }
}

void BrownianMotion<3>::do_initialize(const zisa::array_view<real_t, 4> &u) {
  const zisa::int_t N = u.shape(1);
  auto u_hat = zisa::array<complex_t, 4>(zisa::shape_t<4>(3, N, N, N / 2 + 1),
                                         u.memory_location());
  auto fft = make_fft<3>(u_hat, u);
  initialize(u_hat);
  fft->backward();
  scale(static_cast<real_t>(1. / N), u);
  scale(static_cast<real_t>(1. / N), u);
  scale(static_cast<real_t>(1. / N), u);
}

void BrownianMotion<3>::do_initialize(
    const zisa::array_view<complex_t, 4> &u_hat) {
  const auto init = [&](auto &&u_hat_, real_t H) {
    const zisa::int_t N = u_hat_.shape(1);
    const zisa::int_t N_fourier = N / 2 + 1;
    for (int d = 0; d < 3; ++d) {
      u_hat_(d, 0, 0, 0) = 0;
      for (zisa::int_t k1 = 0; k1 < N_fourier; ++k1) {
        for (zisa::int_t k2 = 0; k2 < N_fourier; ++k2) {
          for (zisa::int_t k3 = 0; k3 < N_fourier; ++k3) {
            if (k1 == 0 && k2 == 0 && k3 == 0) {
              continue;
            }
            const real_t ccc = uniform_.get();
            const real_t ccs = uniform_.get();
            const real_t csc = uniform_.get();
            const real_t css = uniform_.get();
            const real_t scc = uniform_.get();
            const real_t scs = uniform_.get();
            const real_t ssc = uniform_.get();
            const real_t sss = uniform_.get();
            const real_t fac = static_cast<real_t>(N * N * N)
                               / zisa::pow(static_cast<real_t>(
                                               4 * zisa::pi * zisa::pi
                                               * (k1 * k1 + k2 * k2 + k3 * k3)),
                                           (H + 1) / 2);
            u_hat_(d, k1, k2, k3)
                = fac * complex_t(ccc - css - scs - ssc, ccs + csc + scc - sss);
            if (k2 > 0) {
              u_hat_(d, k1, N - k2, k3)
                  = fac
                    * complex_t(ccc + css - scs + ssc, ccs - csc + scc + sss);
            }
            if (k1 > 0) {
              u_hat_(d, N - k1, k2, k3)
                  = fac
                    * complex_t(ccc - css + scs + ssc, ccs + csc - scc + sss);
            }
            if (k1 > 0 && k2 > 0) {
              u_hat_(d, N - k1, N - k2, k3)
                  = fac
                    * complex_t(ccc + css + scs - ssc, ccs - csc - scc - sss);
            }
          }
        }
      }
    }
  };
  const real_t H = hurst_.get();
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    init(u_hat, H);
  } else if (u_hat.memory_location() == zisa::device_type::cuda) {
    auto h_u_hat
        = zisa::array<complex_t, 4>(u_hat.shape(), zisa::device_type::cpu);
    init(h_u_hat, H);
    zisa::copy(u_hat, h_u_hat);
  } else {
    LOG_ERR("Unknown Memory Location");
  }
}

#if AZEBAN_HAS_MPI
void BrownianMotion<3>::do_initialize(const zisa::array_view<real_t, 4> &u,
                                      const Grid<3> &grid,
                                      const Communicator *comm,
                                      void *work_area) {
  const long N = grid.N_phys;
  auto u_hat
      = grid.make_array_fourier(u.shape(0), zisa::device_type::cuda, comm);
  if (u.memory_location() == zisa::device_type::cpu) {
    auto u_device = zisa::array<real_t, 4>(u.shape(), zisa::device_type::cuda);
    auto fft = make_fft_mpi<3>(u_hat, u_device, comm, FFT_BACKWARD, work_area);
    do_initialize(u_hat);
    scale(static_cast<complex_t>(1. / N), u_hat.view());
    scale(static_cast<complex_t>(1. / N), u_hat.view());
    scale(static_cast<complex_t>(1. / N), u_hat.view());
    fft->backward();
    zisa::copy(u, u_device);
  } else if (u.memory_location() == zisa::device_type::cuda) {
    auto fft = make_fft_mpi<3>(u_hat, u, comm, FFT_BACKWARD, work_area);
    do_initialize(u_hat);
    scale(static_cast<complex_t>(1. / N), u_hat.view());
    scale(static_cast<complex_t>(1. / N), u_hat.view());
    scale(static_cast<complex_t>(1. / N), u_hat.view());
    fft->backward();
  } else {
    LOG_ERR("Unknown memory location");
  }
}

void BrownianMotion<3>::do_initialize(
    const zisa::array_view<complex_t, 4> &u_hat,
    const Grid<3> &grid,
    const Communicator *comm,
    void *work_area) {
  ZISA_UNUSED(work_area);
  const auto init = [&](const zisa::array_view<complex_t, 4> &u_hat_host,
                        real_t H) {
    const zisa::int_t N = u_hat.shape(3);
    const zisa::int_t N_fourier = N / 2 + 1;
    const long k_start = grid.i_fourier(0, comm);
    for (int d = 0; d < 3; ++d) {
      for (zisa::int_t i = 0; i < u_hat_host.shape(1); ++i) {
        const zisa::int_t k3 = i + k_start;
        for (zisa::int_t j = 0; j < N_fourier; ++j) {
          const zisa::int_t k2 = j;
          for (zisa::int_t k = 0; k < N_fourier; ++k) {
            const zisa::int_t k1 = k;
            if (k1 == 0 && k2 == 0 && k3 == 0) {
              u_hat_host(d, i, j, k) = 0;
              continue;
            }
            const real_t ccc = uniform_.get();
            const real_t ccs = uniform_.get();
            const real_t csc = uniform_.get();
            const real_t css = uniform_.get();
            const real_t scc = uniform_.get();
            const real_t scs = uniform_.get();
            const real_t ssc = uniform_.get();
            const real_t sss = uniform_.get();
            const real_t fac = static_cast<real_t>(N * N * N)
                               / zisa::pow(static_cast<real_t>(
                                               4 * zisa::pi * zisa::pi
                                               * (k1 * k1 + k2 * k2 + k3 * k3)),
                                           (H + 1) / 2);
            u_hat_host(d, i, j, k)
                = fac * complex_t(ccc - css - scs - ssc, ccs + csc + scc - sss);
            if (k2 > 0) {
              u_hat_host(d, i, N - j, k)
                  = fac
                    * complex_t(ccc + css - scs + ssc, ccs - csc + scc + sss);
            }
            if (k1 > 0) {
              u_hat_host(d, i, j, N - k)
                  = fac
                    * complex_t(ccc - css + scs + ssc, ccs + csc - scc + sss);
            }
            if (k1 > 0 && k2 > 0) {
              u_hat_host(d, i, N - j, N - k)
                  = fac
                    * complex_t(ccc + css + scs - ssc, ccs - csc - scc - sss);
            }
          }
        }
      }
    }
  };
  const real_t H = hurst_.get();
  if (u_hat.memory_location() == zisa::device_type::cpu) {
    init(u_hat, H);
  } else if (u_hat.memory_location() == zisa::device_type::cuda) {
    auto h_u_hat
        = zisa::array<complex_t, 4>(u_hat.shape(), zisa::device_type::cpu);
    init(h_u_hat, H);
    zisa::copy(u_hat, h_u_hat);
  } else {
    LOG_ERR("Unknown Memory Location");
  }
}
#endif

}
