#include <azeban/equations/incompressible_euler_mpi.hpp>
#include <azeban/mpi_types.hpp>
#include <azeban/operations/copy_to_padded.hpp>
#include <azeban/operations/fft_factory.hpp>

namespace azeban {

template <int Dim>
IncompressibleEuler_MPI_Base<Dim>::IncompressibleEuler_MPI_Base(
    const Grid<dim_v> &grid,
    MPI_Comm comm,
    zisa::device_type device,
    bool has_tracer)
    : super(grid),
      comm_(comm),
      device_(device),
      has_tracer_(has_tracer),
      u_hat_pad_({}, nullptr),
      u_yz_({}, nullptr),
      u_yz_pre_({}, nullptr),
      u_yz_comm_({}, nullptr),
      u_yz_trans_({}, nullptr),
      u_yz_trans_pad_({}, nullptr),
      u_xyz_trans_({}, nullptr),
      B_xyz_trans_({}, nullptr) {
  MPI_Comm_rank(comm_, &mpi_rank_);
  MPI_Comm_size(comm_, &mpi_size_);

  size_t ws1_size = 0;
  size_t ws2_size = 0;
  size_t ws_fft_size = 0;

  const int n_vars_u = dim_v + (has_tracer_ ? 1 : 0);
  const int n_vars_B = (dim_v * dim_v + dim_v) / 2 + (has_tracer_ ? dim_v : 0);

  if constexpr (dim_v == 2) {
    fft_u_yz_ = make_fft<dim_v, complex_t>(device, FFT_BACKWARD, false, true);
    fft_u_x_ = make_fft<dim_v, real_t>(device, FFT_BACKWARD, false, true);
    fft_B_yz_ = make_fft<dim_v, complex_t>(device, FFT_FORWARD, false, true);
    fft_B_x_ = make_fft<dim_v, real_t>(device, FFT_FORWARD, false, true);
  } else {
    fft_u_yz_
        = make_fft<dim_v, complex_t>(device, FFT_BACKWARD, false, true, true);
    fft_u_x_ = make_fft<dim_v, real_t>(device, FFT_BACKWARD, false, true, true);
    fft_B_yz_
        = make_fft<dim_v, complex_t>(device, FFT_FORWARD, false, true, true);
    fft_B_x_ = make_fft<dim_v, real_t>(device, FFT_FORWARD, false, true, true);
  }

  const auto shape_u_fourier = grid.shape_fourier(n_vars_u, comm_);
  const auto shape_u_fourier_pad = grid.shape_fourier_pad(n_vars_u, comm_);
  const auto shape_u_phys = grid.shape_phys(n_vars_u, comm_);
  const auto shape_u_phys_pad = grid.shape_phys_pad(n_vars_u, comm_);
  const auto shape_B_fourier = grid.shape_fourier(n_vars_B, comm_);
  const auto shape_B_fourier_pad = grid.shape_fourier_pad(n_vars_B, comm_);
  const auto shape_B_phys = grid.shape_phys(n_vars_B, comm_);
  const auto shape_B_phys_pad = grid.shape_phys_pad(n_vars_B, comm_);

  // Buffer to store padded u_hat in y(z)-directions
  zisa::shape_t<dim_v + 1> shape_u_hat_pad = shape_u_fourier_pad;
  shape_u_hat_pad[1] = shape_u_fourier[1];
  const size_t size_u_hat_pad
      = sizeof(complex_t) * zisa::product(shape_u_hat_pad);
  ws1_size = zisa::max(ws1_size, size_u_hat_pad);

  // Buffer for u_hat fourier transformed in y(z)-directions
  const auto shape_u_yz = fft_u_yz_->shape_u(shape_u_hat_pad);
  const size_t size_u_yz = sizeof(complex_t) * zisa::product(shape_u_yz);
  ws2_size = zisa::max(ws2_size, size_u_yz);

  // Linear buffer ready for MPI_Alltoallv transpose
  const zisa::shape_t<1> shape_u_yz_pre(zisa::product(shape_u_yz));
  const size_t size_u_yz_pre
      = sizeof(complex_t) * zisa::product(shape_u_yz_pre);
  ws1_size = zisa::max(ws1_size, size_u_yz_pre);

  // Received data after MPI_Alltoallv
  zisa::shape_t<1> shape_u_yz_comm(n_vars_u);
  shape_u_yz_comm[0] *= grid_.N_fourier;
  shape_u_yz_comm[0] *= grid_.N_phys_pad / mpi_size_
                        + (zisa::integer_cast<zisa::int_t>(mpi_rank_)
                           < grid_.N_phys_pad % mpi_size_);
  if (dim_v == 3) {
    shape_u_yz_comm[0] *= grid_.N_phys_pad;
  }
  const size_t size_u_yz_comm
      = sizeof(complex_t) * zisa::product(shape_u_yz_comm);
  ws2_size = zisa::max(ws2_size, size_u_yz_comm);

  // Transposed partially fourier transformed data
  zisa::shape_t<dim_v + 1> shape_u_yz_trans;
  shape_u_yz_trans[0] = n_vars_u;
  shape_u_yz_trans[1] = grid_.N_phys_pad / mpi_size_
                        + (zisa::integer_cast<zisa::int_t>(mpi_rank_)
                           < grid_.N_phys_pad % mpi_size_);
  shape_u_yz_trans[2] = dim_v == 2 ? grid_.N_fourier : grid_.N_phys_pad;
  if (dim_v == 3) {
    shape_u_yz_trans[3] = grid_.N_fourier;
  }
  const size_t size_u_yz_trans
      = sizeof(complex_t) * zisa::product(shape_u_yz_trans);
  ws1_size = zisa::max(ws1_size, size_u_yz_trans);

  // Transposed data padded in the x-direction
  zisa::shape_t<dim_v + 1> shape_u_yz_trans_pad;
  shape_u_yz_trans_pad[0] = n_vars_u;
  shape_u_yz_trans_pad[1] = grid_.N_phys_pad / mpi_size_
                            + (zisa::integer_cast<zisa::int_t>(mpi_rank_)
                               < grid_.N_phys_pad % mpi_size_);
  shape_u_yz_trans_pad[2] = dim_v == 2 ? grid_.N_fourier_pad : grid_.N_phys_pad;
  if (dim_v == 3) {
    shape_u_yz_trans_pad[3] = grid_.N_fourier_pad;
  }
  const size_t size_u_yz_trans_pad
      = sizeof(complex_t) * zisa::product(shape_u_yz_trans_pad);
  ws2_size = zisa::max(ws2_size, size_u_yz_trans_pad);

  // Transposed fully Fourier Transformed data
  const zisa::shape_t<dim_v + 1> shape_u_xyz_trans = shape_u_phys_pad;
  const size_t size_u_xyz_trans
      = sizeof(real_t) * zisa::product(shape_u_xyz_trans);
  ws1_size = zisa::max(ws1_size, size_u_xyz_trans);

  // B in real space
  const zisa::shape_t<dim_v + 1> shape_B_xyz_trans = shape_B_phys_pad;
  const size_t size_B_xyz_trans
      = sizeof(real_t) * zisa::product(shape_B_xyz_trans);
  ws2_size = zisa::max(ws2_size, size_B_xyz_trans);

  // Allocate Workspaces
  ws1_ = Workspace(ws1_size, device);
  ws2_ = Workspace(ws2_size, device);

  // Generate views onto workspaces
  u_hat_pad_ = ws1_.get_view<complex_t>(0, shape_u_hat_pad);
  u_yz_ = ws2_.get_view<complex_t>(0, shape_u_yz);
  u_yz_pre_ = ws1_.get_view<complex_t>(0, shape_u_yz_pre);
  u_yz_comm_ = ws2_.get_view<complex_t>(0, shape_u_yz_comm);
  u_yz_trans_ = ws1_.get_view<complex_t>(0, shape_u_yz_trans);
  u_yz_trans_pad_ = ws2_.get_view<complex_t>(0, shape_u_yz_trans_pad);
  u_xyz_trans_ = ws1_.get_view<real_t>(0, shape_u_xyz_trans);
  B_xyz_trans_ = ws2_.get_view<real_t>(0, shape_B_xyz_trans);

  // Initialize Fourier Transforms
  fft_u_yz_->initialize(u_hat_pad_, u_yz_, false);
  const size_t size_fft_u_yz = fft_u_yz_->get_work_area_size();
  ws_fft_size = zisa::max(ws_fft_size, size_fft_u_yz);
  fft_u_x_->initialize(u_yz_trans_pad_, u_xyz_trans_, false);
  const size_t size_fft_u_x = fft_u_x_->get_work_area_size();
  ws_fft_size = zisa::max(ws_fft_size, size_fft_u_x);

  ws_fft_ = Workspace(ws_fft_size, device);

  fft_u_yz_->set_work_area(ws_fft_.get());
  fft_u_x_->set_work_area(ws_fft_.get());
  fft_B_yz_->set_work_area(ws_fft_.get());
  fft_B_x_->set_work_area(ws_fft_.get());
}

template <int Dim>
void *IncompressibleEuler_MPI_Base<Dim>::get_fft_work_area() {
  return ws_fft_.get();
}

template <>
zisa::array_view<complex_t, 2> IncompressibleEuler_MPI_Base<2>::component(
    const zisa::array_view<complex_t, 3> &arr, int dim) {
  zisa::shape_t<2> shape{arr.shape(1), arr.shape(2)};
  return {shape, arr.raw() + dim * zisa::product(shape), arr.memory_location()};
}

template <>
zisa::array_const_view<complex_t, 2> IncompressibleEuler_MPI_Base<2>::component(
    const zisa::array_const_view<complex_t, 3> &arr, int dim) {
  zisa::shape_t<2> shape{arr.shape(1), arr.shape(2)};
  return {shape, arr.raw() + dim * zisa::product(shape), arr.memory_location()};
}

template <>
zisa::array_view<complex_t, 2>
IncompressibleEuler_MPI_Base<2>::component(zisa::array<complex_t, 3> &arr,
                                           int dim) {
  zisa::shape_t<2> shape{arr.shape(1), arr.shape(2)};
  return {shape, arr.raw() + dim * zisa::product(shape), arr.device()};
}

template <>
zisa::array_const_view<complex_t, 2>
IncompressibleEuler_MPI_Base<2>::component(const zisa::array<complex_t, 3> &arr,
                                           int dim) {
  zisa::shape_t<2> shape{arr.shape(1), arr.shape(2)};
  return {shape, arr.raw() + dim * zisa::product(shape), arr.device()};
}

template <>
zisa::array_view<complex_t, 3> IncompressibleEuler_MPI_Base<3>::component(
    const zisa::array_view<complex_t, 4> &arr, int dim) {
  zisa::shape_t<3> shape{arr.shape(1), arr.shape(2), arr.shape(3)};
  return {shape, arr.raw() + dim * zisa::product(shape), arr.memory_location()};
}

template <>
zisa::array_const_view<complex_t, 3> IncompressibleEuler_MPI_Base<3>::component(
    const zisa::array_const_view<complex_t, 4> &arr, int dim) {
  zisa::shape_t<3> shape{arr.shape(1), arr.shape(2), arr.shape(3)};
  return {shape, arr.raw() + dim * zisa::product(shape), arr.memory_location()};
}

template <>
zisa::array_view<complex_t, 3>
IncompressibleEuler_MPI_Base<3>::component(zisa::array<complex_t, 4> &arr,
                                           int dim) {
  zisa::shape_t<3> shape{arr.shape(1), arr.shape(2), arr.shape(3)};
  return {shape, arr.raw() + dim * zisa::product(shape), arr.device()};
}

template <>
zisa::array_const_view<complex_t, 3>
IncompressibleEuler_MPI_Base<3>::component(const zisa::array<complex_t, 4> &arr,
                                           int dim) {
  zisa::shape_t<3> shape{arr.shape(1), arr.shape(2), arr.shape(3)};
  return {shape, arr.raw() + dim * zisa::product(shape), arr.device()};
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_B_hat(
    const zisa::array_const_view<complex_t, dim_v + 1> &u_hat) {
  compute_u_hat_pad(u_hat);
  compute_u_yz();
  compute_u_yz_pre();
  compute_u_yz_comm();
  compute_u_yz_trans();
  compute_u_yz_trans_pad();
  compute_u_xyz_trans();
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_u_hat_pad(
    const zisa::array_const_view<complex_t, dim_v + 1> &u_hat) {
  for (zisa::int_t d = 0; d < u_hat.shape(0); ++d) {
    if constexpr (dim_v == 2) {
      copy_to_padded(
          false, true, 0, component(u_hat_pad_, d), component(u_hat, d));
    } else {
      copy_to_padded(
          false, true, true, 0, component(u_hat_pad_, d), component(u_hat, d));
    }
  }
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_u_yz() {
  fft_u_yz_->backward();
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_u_yz_pre() {
  if (device_ == zisa::device_type::cpu) {
    compute_u_yz_pre_cpu();
  }
#if ZISA_HAS_CUDA
  else if (device_ == zisa::device_type::cuda) {
    // TODO: Implement
    LOG_ERR("Not yet implemented");
  }
#endif
  else {
    LOG_ERR("Unsupported device");
  }
}

template <>
void IncompressibleEuler_MPI_Base<2>::compute_u_yz_pre_cpu() {
  const zisa::int_t ndim = u_yz_.shape(0);
  const zisa::int_t N_loc = u_yz_.shape(1);
  size_t j_offset = 0;
  for (int r = 0; r < mpi_size_; ++r) {
    const zisa::int_t N_r
        = grid_.N_phys_pad / mpi_size_
          + (zisa::integer_cast<zisa::int_t>(r) < grid_.N_phys_pad % mpi_size_);
    zisa::array_view<complex_t, 3> out_view({ndim, N_r, N_loc},
                                            u_yz_pre_.raw()
                                                + j_offset * N_loc * ndim,
                                            zisa::device_type::cpu);
    for (zisa::int_t d = 0; d < ndim; ++d) {
      for (zisa::int_t i = 0; i < N_loc; ++i) {
        for (zisa::int_t j = 0; j < N_r; ++j) {
          out_view(d, j, i) = u_yz_(d, i, j_offset + j);
        }
      }
    }
    j_offset += N_r;
  }
}

template <>
void IncompressibleEuler_MPI_Base<3>::compute_u_yz_pre_cpu() {
  // TODO: Implement
  LOG_ERR("Not yet implemented");
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_u_yz_comm() {
  const zisa::int_t ndims = dim_v + (has_tracer_ ? 1 : 0);
  auto sendcnts = std::make_unique<int[]>(mpi_size_);
  auto sdispls = std::make_unique<int[]>(mpi_size_ + 1);
  auto recvcnts = std::make_unique<int[]>(mpi_size_);
  auto rdispls = std::make_unique<int[]>(mpi_size_ + 1);
  sdispls[0] = 0;
  rdispls[0] = 0;
  const zisa::int_t N_loc_pre = grid_.N_fourier_pad / mpi_size_
                                + (zisa::integer_cast<zisa::int_t>(mpi_rank_)
                                   < grid_.N_fourier_pad % mpi_size_);
  const zisa::int_t N_loc_post = grid_.N_phys_pad / mpi_size_
                                 + (zisa::integer_cast<zisa::int_t>(mpi_rank_)
                                    < grid_.N_phys_pad % mpi_size_);
  for (int r = 0; r < mpi_size_; ++r) {
    const zisa::int_t N_r_pre = grid_.N_fourier_pad / mpi_size_
                                + (zisa::integer_cast<zisa::int_t>(r)
                                   < grid_.N_fourier_pad % mpi_size_);
    const zisa::int_t N_r_post
        = grid_.N_phys_pad / mpi_size_
          + (zisa::integer_cast<zisa::int_t>(r) < grid_.N_phys_pad % mpi_size_);
    sendcnts[r]
        = ndims * N_loc_pre * N_r_post * (dim_v == 3 ? grid_.N_phys_pad : 1);
    recvcnts[r]
        = ndims * N_loc_post * N_r_pre * (dim_v == 3 ? grid_.N_phys_pad : 1);
    sdispls[r + 1] = sdispls[r] + sendcnts[r];
    rdispls[r + 1] = rdispls[r] + recvcnts[r];
  }

  if (device_ == zisa::device_type::cpu) {
    MPI_Alltoallv(u_yz_pre_.raw(),
                  sendcnts.get(),
                  sdispls.get(),
                  mpi_type<complex_t>(),
                  u_yz_comm_.raw(),
                  recvcnts.get(),
                  rdispls.get(),
                  mpi_type<complex_t>(),
                  comm_);
  }
#if ZISA_HAS_CUDA
  else if (device_ == zisa::device_type::cuda) {
    // TODO: Implement
    LOG_ERR("Not yet implemented");
  }
#endif
  else {
    LOG_ERR("Unsupported device");
  }
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_u_yz_trans() {
  if (device_ == zisa::device_type::cpu) {
    compute_u_yz_trans_cpu();
  }
#if ZISA_HAS_CUDA
  else if (device_ == zisa::device_type::cuda) {
    // TODO: Implement
    LOG_ERR("Not yet implemented");
  }
#endif
  else {
    LOG_ERR("Unsupported device");
  }
}

template <>
void IncompressibleEuler_MPI_Base<2>::compute_u_yz_trans_cpu() {
  const zisa::int_t ndim = dim_v + (has_tracer_ ? 1 : 0);
  const zisa::int_t N_loc = grid_.N_phys_pad / mpi_size_
                            + (zisa::integer_cast<zisa::int_t>(mpi_rank_)
                               < grid_.N_phys_pad % mpi_size_);
  zisa::int_t j_offset = 0;
  for (int r = 0; r < mpi_size_; ++r) {
    const zisa::int_t N_r = grid_.N_fourier_pad / mpi_size_
                            + (zisa::integer_cast<zisa::int_t>(r)
                               < grid_.N_fourier_pad % mpi_size_);
    zisa::array_view<complex_t, 3> in_view({ndim, N_loc, N_r},
                                           u_yz_pre_.raw()
                                               + j_offset * N_loc * ndim,
                                           zisa::device_type::cpu);
    for (zisa::int_t d = 0; d < ndim; ++d) {
      for (zisa::int_t i = 0; i < N_loc; ++i) {
        for (zisa::int_t j = 0; j < N_r; ++j) {
          u_yz_trans_(d, i, j_offset + j) = in_view(d, i, j);
        }
      }
    }
    j_offset += N_r;
  }
}

template <>
void IncompressibleEuler_MPI_Base<3>::compute_u_yz_trans_cpu() {
  // TODO: Implement
  LOG_ERR("Not yet implemented");
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_u_yz_trans_pad() {
  for (zisa::int_t d = 0; d < u_yz_trans_.shape(0); ++d) {
    if constexpr (dim_v == 2) {
      copy_to_padded(false,
                     true,
                     1,
                     component(u_yz_trans_pad_, d),
                     component(u_yz_trans_, d));
    } else {
      copy_to_padded(false,
                     false,
                     true,
                     2,
                     component(u_yz_trans_pad_, d),
                     component(u_yz_trans_, d));
    }
  }
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_u_xyz_trans() {
  fft_u_x_->backward();
}

template class IncompressibleEuler_MPI_Base<2>;
template class IncompressibleEuler_MPI_Base<3>;

}
