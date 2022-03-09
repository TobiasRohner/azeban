#include <azeban/equations/advection_functions.hpp>
#include <azeban/equations/incompressible_euler_functions.hpp>
#include <azeban/equations/incompressible_euler_mpi.hpp>
#include <azeban/mpi/mpi_types.hpp>
#include <azeban/operations/copy_from_padded.hpp>
#include <azeban/operations/copy_to_padded.hpp>
#include <azeban/operations/fft_factory.hpp>
#include <sstream>
#if ZISA_HAS_CUDA
#include <azeban/cuda/equations/incompressible_euler_cuda.hpp>
#include <azeban/memory/mapped_array.hpp>
#endif

namespace azeban {

template <int Dim>
IncompressibleEuler_MPI_Base<Dim>::IncompressibleEuler_MPI_Base(
    const Grid<dim_v> &grid,
    const Communicator *comm,
    zisa::device_type device,
    bool has_tracer)
    : super(grid),
      comm_(comm),
      device_(device),
      has_tracer_(has_tracer),
      B_hat_({}, nullptr),
      mpi_rank_(comm->rank()),
      mpi_size_(comm->size()),
      u_hat_pad_({}, nullptr),
      u_yz_({}, nullptr),
      trans_u_sendbuf_({}, nullptr),
      u_yz_trans_({}, nullptr),
      u_yz_trans_pad_({}, nullptr),
      u_xyz_trans_({}, nullptr),
      B_xyz_trans_({}, nullptr),
      B_yz_trans_pad_({}, nullptr),
      B_yz_trans_({}, nullptr),
      trans_B_sendbuf_({}, nullptr),
      B_yz_({}, nullptr),
      B_hat_pad_({}, nullptr) {
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
    fft_u_x_
        = make_fft<dim_v, real_t>(device, FFT_BACKWARD, false, false, true);
    fft_B_yz_
        = make_fft<dim_v, complex_t>(device, FFT_FORWARD, false, true, true);
    fft_B_x_ = make_fft<dim_v, real_t>(device, FFT_FORWARD, false, false, true);
  }

  const auto shape_u_fourier = grid.shape_fourier(n_vars_u, comm_);
  const auto shape_u_fourier_pad = grid.shape_fourier_pad(n_vars_u, comm_);
  const auto shape_u_phys_pad = grid.shape_phys_pad(n_vars_u, comm_);
  const auto shape_B_fourier = grid.shape_fourier(n_vars_B, comm_);
  const auto shape_B_fourier_pad = grid.shape_fourier_pad(n_vars_B, comm_);
  const auto shape_B_phys_pad = grid.shape_phys_pad(n_vars_B, comm_);

  // Buffer to store padded u_hat in y(z)-directions
  zisa::shape_t<dim_v + 1> shape_u_hat_pad = shape_u_fourier_pad;
  shape_u_hat_pad[1] = shape_u_fourier[1];
  const size_t size_u_hat_pad
      = sizeof(complex_t) * zisa::product(shape_u_hat_pad);
  ws1_size = zisa::max(ws1_size, size_u_hat_pad);

  // Buffer for u_hat fourier transformed in y(z)-directions
  const auto shape_u_yz = shape_u_hat_pad;
  const size_t size_u_yz = sizeof(complex_t) * zisa::product(shape_u_yz);
  ws2_size = zisa::max(ws2_size, size_u_yz);

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
  ws2_size = zisa::max(ws2_size, size_u_yz_trans);

  // Buffer for transposing u_yz_
  transpose_u_ = std::make_shared<Transpose<dim_v>>(
      comm, shape_u_yz, shape_u_yz_trans, device);
  const auto trans_u_buf_shape = transpose_u_->buffer_shape();
  const size_t size_trans_u_buf
      = sizeof(complex_t) * zisa::product(trans_u_buf_shape);
  ws1_size = zisa::max(ws1_size, size_trans_u_buf);

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
  ws1_size = zisa::max(ws1_size, size_u_yz_trans_pad);

  // Transposed fully Fourier Transformed data
  const zisa::shape_t<dim_v + 1> shape_u_xyz_trans = shape_u_phys_pad;
  const size_t size_u_xyz_trans
      = sizeof(real_t) * zisa::product(shape_u_xyz_trans);
  ws2_size = zisa::max(ws2_size, size_u_xyz_trans);

  // B in real space
  const zisa::shape_t<dim_v + 1> shape_B_xyz_trans = shape_B_phys_pad;
  const size_t size_B_xyz_trans
      = sizeof(real_t) * zisa::product(shape_B_xyz_trans);
  ws1_size = zisa::max(ws1_size, size_B_xyz_trans);

  // B Fourier transfomed in the x-direction
  zisa::shape_t<dim_v + 1> shape_B_yz_trans_pad = shape_B_xyz_trans;
  shape_B_yz_trans_pad[dim_v] = shape_B_yz_trans_pad[dim_v] / 2 + 1;
  const size_t size_B_yz_trans_pad
      = sizeof(complex_t) * zisa::product(shape_B_yz_trans_pad);
  ws2_size = zisa::max(ws2_size, size_B_yz_trans_pad);

  // Unpadded B_yz
  zisa::shape_t<dim_v + 1> shape_B_yz_trans = shape_B_yz_trans_pad;
  shape_B_yz_trans[dim_v] = grid_.N_fourier;
  const size_t size_B_yz_trans
      = sizeof(complex_t) * zisa::product(shape_B_yz_trans);
  ws1_size = zisa::max(ws1_size, size_B_yz_trans);

  // Buffer for B_hat fourier transformed in y(z)-directions
  zisa::shape_t<dim_v + 1> shape_B_yz = shape_B_fourier_pad;
  shape_B_yz[1] = shape_B_fourier[1];
  const size_t size_B_yz = sizeof(complex_t) * zisa::product(shape_B_yz);
  ws1_size = zisa::max(ws1_size, size_B_yz);

  // Buffer for transposing B
  transpose_B_ = std::make_shared<Transpose<dim_v>>(
      comm, shape_B_yz_trans, shape_B_yz, device);
  const auto trans_B_buf_shape = transpose_B_->buffer_shape();
  const size_t size_trans_B_buf
      = sizeof(complex_t) * zisa::product(trans_B_buf_shape);
  ws2_size = zisa::max(ws2_size, size_trans_B_buf);

  // Buffer to store padded B_hat in y(z)-directions
  zisa::shape_t<dim_v + 1> shape_B_hat_pad = shape_B_fourier_pad;
  shape_B_hat_pad[1] = shape_B_fourier[1];
  const size_t size_B_hat_pad
      = sizeof(complex_t) * zisa::product(shape_B_hat_pad);
  ws2_size = zisa::max(ws2_size, size_B_hat_pad);

  // Buffer to store B_hat
  const auto shape_B_hat = shape_B_fourier;
  const size_t size_B_hat = sizeof(complex_t) * zisa::product(shape_B_hat);
  ws1_size = zisa::max(ws1_size, size_B_hat);

  // Allocate Workspaces
  ws1_ = Workspace(ws1_size, device);
  ws2_ = Workspace(ws2_size, device);

  // Generate views onto workspaces
  u_hat_pad_ = ws1_.get_view<complex_t>(0, shape_u_hat_pad);
  u_yz_ = ws2_.get_view<complex_t>(0, shape_u_yz);
  trans_u_sendbuf_ = ws1_.get_view<complex_t>(0, trans_u_buf_shape);
  if (device == zisa::device_type::cpu) {
    trans_u_recvbuf_ = zisa::array<complex_t, dim_v + 2>(
        trans_u_buf_shape, zisa::device_type::cpu);
  }
#if ZISA_HAS_CUDA
  else if (device == zisa::device_type::cuda) {
    trans_u_recvbuf_ = mapped_array<complex_t, dim_v + 2>(trans_u_buf_shape);
  }
#endif
  else {
    LOG_ERR("Unsupported device");
  }
  u_yz_trans_ = ws2_.get_view<complex_t>(0, shape_u_yz_trans);
  u_yz_trans_pad_ = ws1_.get_view<complex_t>(0, shape_u_yz_trans_pad);
  u_xyz_trans_ = ws2_.get_view<real_t>(0, shape_u_xyz_trans);
  B_xyz_trans_ = ws1_.get_view<real_t>(0, shape_B_xyz_trans);
  B_yz_trans_pad_ = ws2_.get_view<complex_t>(0, shape_B_yz_trans_pad);
  B_yz_trans_ = ws1_.get_view<complex_t>(0, shape_B_yz_trans);
  trans_B_sendbuf_ = ws2_.get_view<complex_t>(0, trans_B_buf_shape);
  if (device == zisa::device_type::cpu) {
    trans_B_recvbuf_ = zisa::array<complex_t, dim_v + 2>(
        trans_B_buf_shape, zisa::device_type::cpu);
  }
#if ZISA_HAS_CUDA
  else if (device == zisa::device_type::cuda) {
    trans_B_recvbuf_ = mapped_array<complex_t, dim_v + 2>(trans_B_buf_shape);
  }
#endif
  else {
    LOG_ERR("Unsupported device");
  }
  B_yz_ = ws1_.get_view<complex_t>(0, shape_B_yz);
  B_hat_pad_ = ws2_.get_view<complex_t>(0, shape_B_hat_pad);
  B_hat_ = ws1_.get_view<complex_t>(0, shape_B_hat);

  std::stringstream ss;
  ss << "rank " << mpi_rank_ << "\n"
     << "u_hat_pad_.shape()       = " << u_hat_pad_.shape() << "\n"
     << "u_yz_.shape()            = " << u_yz_.shape() << "\n"
     << "trans_u_sendbuf_.shape() = " << trans_u_sendbuf_.shape() << "\n"
     << "trans_u_recvbuf_.shape() = " << trans_u_recvbuf_.shape() << "\n"
     << "u_yz_trans_.shape()      = " << u_yz_trans_.shape() << "\n"
     << "u_yz_trans_pad_.shape()  = " << u_yz_trans_pad_.shape() << "\n"
     << "u_xyz_trans_.shape()     = " << u_xyz_trans_.shape() << "\n"
     << "B_xyz_trans_.shape()     = " << B_xyz_trans_.shape() << "\n"
     << "B_yz_trans_pad_.shape()  = " << B_yz_trans_pad_.shape() << "\n"
     << "B_yz_trans_.shape()      = " << B_yz_trans_.shape() << "\n"
     << "trans_B_sendbuf_.shape() = " << trans_B_sendbuf_.shape() << "\n"
     << "trans_B_recvbuf_.shape() = " << trans_B_recvbuf_.shape() << "\n"
     << "B_yz_.shape()            = " << B_yz_.shape() << "\n"
     << "B_hat_pad_.shape()       = " << B_hat_pad_.shape() << "\n"
     << "B_hat_.shape()           = " << B_hat_.shape() << std::endl;
  for (int r = 0; r < mpi_size_; ++r) {
    if (mpi_rank_ == r) {
      std::cout << ss.str();
    }
    MPI_Barrier(comm_->get_mpi_comm());
  }

  // Initialize Fourier Transforms
  fft_u_yz_->initialize(u_hat_pad_, u_yz_, false);
  const size_t size_fft_u_yz = fft_u_yz_->get_work_area_size();
  ws_fft_size = zisa::max(ws_fft_size, size_fft_u_yz);
  fft_u_x_->initialize(u_yz_trans_pad_, u_xyz_trans_, false);
  const size_t size_fft_u_x = fft_u_x_->get_work_area_size();
  ws_fft_size = zisa::max(ws_fft_size, size_fft_u_x);
  fft_B_x_->initialize(B_yz_trans_pad_, B_xyz_trans_, false);
  const size_t size_fft_B_x = fft_B_x_->get_work_area_size();
  ws_fft_size = zisa::max(ws_fft_size, size_fft_B_x);
  fft_B_yz_->initialize(B_hat_pad_, B_yz_, false);
  const size_t size_fft_B_yz = fft_B_yz_->get_work_area_size();
  ws_fft_size = zisa::max(ws_fft_size, size_fft_B_yz);

  ws_fft_ = Workspace(ws_fft_size, device);

  fft_u_yz_->set_work_area(ws_fft_.get());
  fft_u_x_->set_work_area(ws_fft_.get());
  fft_B_yz_->set_work_area(ws_fft_.get());
  fft_B_x_->set_work_area(ws_fft_.get());

  // Initialize the transposes
  transpose_u_->set_from_array(u_yz_);
  transpose_u_->set_send_buffer(trans_u_sendbuf_);
  transpose_u_->set_recv_buffer(trans_u_recvbuf_);
  transpose_u_->set_to_array(u_yz_trans_);
  transpose_B_->set_from_array(B_yz_trans_);
  transpose_B_->set_send_buffer(trans_B_sendbuf_);
  transpose_B_->set_recv_buffer(trans_B_recvbuf_);
  transpose_B_->set_to_array(B_yz_);
}

template <int Dim>
void *IncompressibleEuler_MPI_Base<Dim>::get_fft_work_area() const {
  return const_cast<void *>(ws_fft_.get());
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
void IncompressibleEuler_MPI_Base<Dim>::computeBhat(
    const zisa::array_const_view<complex_t, dim_v + 1> &u_hat) {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::computeBhat");
  compute_u_hat_pad(u_hat);
  compute_u_yz();
  compute_u_yz_trans();
  compute_u_yz_trans_pad();
  compute_u_xyz_trans();
  compute_B_xyz_trans();
  compute_B_yz_trans_pad();
  compute_B_yz_trans();
  compute_B_yz();
  compute_B_hat_pad();
  compute_B_hat();
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::computeBhat");
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_u_hat_pad(
    const zisa::array_const_view<complex_t, dim_v + 1> &u_hat) {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::compute_u_hat_pad");
  for (zisa::int_t d = 0; d < u_hat.shape(0); ++d) {
    if constexpr (dim_v == 2) {
      copy_to_padded(
          false, true, 0, component(u_hat_pad_, d), component(u_hat, d));
    } else {
      copy_to_padded(
          false, true, true, 0, component(u_hat_pad_, d), component(u_hat, d));
    }
  }
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::compute_u_hat_pad");
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_u_yz() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::compute_u_yz");
  fft_u_yz_->backward();
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::compute_u_yz");
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_u_yz_trans() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::compute_u_yz_trans");
  transpose_u_->eval();
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::compute_u_yz_trans");
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_u_yz_trans_pad() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::compute_u_yz_trans_pad");
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
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::compute_u_yz_trans_pad");
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_u_xyz_trans() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::compute_u_xyz_trans");
  fft_u_x_->backward();
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::compute_u_xyz_trans");
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_B_xyz_trans() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::compute_B_xyz_trans");
  if (device_ == zisa::device_type::cpu) {
    compute_B_xyz_trans_cpu();
  }
#if ZISA_HAS_CUDA
  else if (device_ == zisa::device_type::cuda) {
    if (has_tracer_) {
      incompressible_euler_compute_B_tracer_cuda<Dim>(
          B_xyz_trans_, u_xyz_trans_, grid_);
    } else {
      incompressible_euler_compute_B_cuda<Dim>(
          B_xyz_trans_, u_xyz_trans_, grid_);
    }
  }
#endif
  else {
    LOG_ERR("Unsupported device");
  }
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::compute_B_xyz_trans");
}

template <>
void IncompressibleEuler_MPI_Base<2>::compute_B_xyz_trans_cpu() {
  const real_t norm = 1.0 / (grid_.N_phys * grid_.N_phys_pad);
  const unsigned stride
      = zisa::product(u_xyz_trans_.shape()) / u_xyz_trans_.shape(0);
  for (zisa::int_t i = 0; i < u_xyz_trans_.shape(1); ++i) {
    for (zisa::int_t j = 0; j < u_xyz_trans_.shape(2); ++j) {
      const unsigned idx = i * u_xyz_trans_.shape(2) + j;
      const real_t u1 = norm * u_xyz_trans_(0, i, j);
      const real_t u2 = norm * u_xyz_trans_(1, i, j);
      incompressible_euler_2d_compute_B(
          stride, idx, u1, u2, B_xyz_trans_.raw());
      if (has_tracer_) {
        const real_t rho = norm * u_xyz_trans_(2, i, j);
        advection_2d_compute_B(
            stride, idx, rho, u1, u2, B_xyz_trans_.raw() + 3 * stride);
      }
    }
  }
}

template <>
void IncompressibleEuler_MPI_Base<3>::compute_B_xyz_trans_cpu() {
  const real_t norm = 1.0
                      / (zisa::pow<dim_v>(zisa::sqrt(grid_.N_phys))
                         * zisa::pow<dim_v>(zisa::sqrt(grid_.N_phys_pad)));
  const unsigned stride
      = zisa::product(u_xyz_trans_.shape()) / u_xyz_trans_.shape(0);
  for (zisa::int_t i = 0; i < u_xyz_trans_.shape(1); ++i) {
    for (zisa::int_t j = 0; j < u_xyz_trans_.shape(2); ++j) {
      for (zisa::int_t k = 0; k < u_xyz_trans_.shape(3); ++k) {
        const unsigned idx = i * u_xyz_trans_.shape(2) * u_xyz_trans_.shape(3)
                             + j * u_xyz_trans_.shape(3) + k;
        const real_t u1 = norm * u_xyz_trans_(0, i, j, k);
        const real_t u2 = norm * u_xyz_trans_(1, i, j, k);
        const real_t u3 = norm * u_xyz_trans_(2, i, j, k);
        incompressible_euler_3d_compute_B(
            stride, idx, u1, u2, u3, B_xyz_trans_.raw());
        if (has_tracer_) {
          const real_t rho = norm * u_xyz_trans_(3, i, j, k);
          advection_3d_compute_B(
              stride, idx, rho, u1, u2, u3, B_xyz_trans_.raw() + 6 * stride);
        }
      }
    }
  }
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_B_yz_trans_pad() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::compute_B_xyz_trans_pad");
  fft_B_x_->forward();
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::compute_B_xyz_trans_pad");
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_B_yz_trans() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::compute_B_xyz_trans");
  for (zisa::int_t d = 0; d < B_yz_trans_pad_.shape(0); ++d) {
    if constexpr (dim_v == 2) {
      copy_from_padded(false,
                       true,
                       1,
                       component(B_yz_trans_, d),
                       component(B_yz_trans_pad_, d));
    } else {
      copy_from_padded(false,
                       false,
                       true,
                       2,
                       component(B_yz_trans_, d),
                       component(B_yz_trans_pad_, d));
    }
  }
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::compute_B_xyz_trans");
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_B_yz() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::compute_B_yz");
  transpose_B_->eval();
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::compute_B_yz");
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_B_hat_pad() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::compute_B__hat_pad");
  fft_B_yz_->forward();
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::compute_B__hat_pad");
}

template <int Dim>
void IncompressibleEuler_MPI_Base<Dim>::compute_B_hat() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::compute_B__hat");
  for (zisa::int_t d = 0; d < B_hat_pad_.shape(0); ++d) {
    if constexpr (dim_v == 2) {
      copy_from_padded(
          false, true, 0, component(B_hat_, d), component(B_hat_pad_, d));
    } else {
      copy_from_padded(
          false, true, true, 0, component(B_hat_, d), component(B_hat_pad_, d));
    }
  }
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::compute_B__hat");
}

template class IncompressibleEuler_MPI_Base<2>;
template class IncompressibleEuler_MPI_Base<3>;

}
