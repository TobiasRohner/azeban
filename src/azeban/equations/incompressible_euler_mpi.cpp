#if AZEBAN_HAS_MPI

#include <azeban/equations/incompressible_euler_mpi.hpp>
#include <azeban/mpi_types.hpp>
#include <vector>

namespace azeban {

template <>
IncompressibleEuler_MPI_Base<2>::IncompressibleEuler_MPI_Base(
    const Grid<2> &grid, MPI_Comm comm, bool has_tracer)
    : super(grid), comm_(comm), has_tracer_(has_tracer) {
  const zisa::shape_t<3> shape_unpad = grid_.shape_fourier(1, comm);
  const zisa::shape_t<3> shape_pad = grid_.shape_fourier_pad(1, comm);
  const zisa::int_t n_var_u = 2 + (has_tracer ? 1 : 0);
  const zisa::int_t n_var_B = 3 + (has_tracer ? 2 : 0);
  u_hat_partial_pad_ = zisa::array<complex_t, 3>(
      zisa::shape_t<3>(n_var_u, shape_pad[1], shape_unpad[2]),
      zisa::device_type::cpu);
  h_u_hat_pad_
      = grid_.make_array_fourier_pad(n_var_u, zisa::device_type::cpu, comm);
  d_u_hat_pad_
      = grid_.make_array_fourier_pad(n_var_u, zisa::device_type::cuda, comm);
  u_pad_ = grid_.make_array_phys_pad(n_var_u, zisa::device_type::cuda, comm);
  B_pad_ = grid_.make_array_phys_pad(n_var_B, zisa::device_type::cuda, comm);
  d_B_hat_pad_
      = grid_.make_array_fourier_pad(n_var_B, zisa::device_type::cuda, comm);
  h_B_hat_pad_
      = grid_.make_array_fourier_pad(n_var_B, zisa::device_type::cpu, comm);
  B_hat_partial_pad_ = zisa::array<complex_t, 3>(
      zisa::shape_t<3>(n_var_B, shape_pad[1], shape_unpad[2]),
      zisa::device_type::cpu);
  B_hat_ = grid_.make_array_fourier(n_var_B, zisa::device_type::cpu, comm);
  fft_u_ = make_fft_mpi<2>(d_u_hat_pad_, u_pad_, comm, FFT_BACKWARD);
  fft_B_ = make_fft_mpi<2>(d_B_hat_pad_, B_pad_, comm, FFT_FORWARD);
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
void IncompressibleEuler_MPI_Base<2>::computeB() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::computeB");
  if (has_tracer_) {
    incompressible_euler_compute_B_tracer_cuda<2>(
        fft_B_->u(), fft_u_->u(), grid_);
  } else {
    incompressible_euler_compute_B_cuda<2>(fft_B_->u(), fft_u_->u(), grid_);
  }
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::computeB");
}

template <>
void IncompressibleEuler_MPI_Base<2>::pad_u_hat(
    const zisa::array_const_view<complex_t, 3> &u_hat) {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::pad_u_hat");
  int rank, size;
  MPI_Comm_rank(comm_, &rank);
  MPI_Comm_size(comm_, &size);
  std::vector<MPI_Request> send_reqs;
  std::vector<MPI_Request> recv_reqs;
  const zisa::int_t n_vars = u_hat.shape(0);
  const zisa::int_t i0 = grid_.i_fourier(0, rank, comm_);
  const zisa::int_t i1 = grid_.i_fourier(0, rank + 1, comm_);
  const zisa::int_t i0_pad = grid_.i_fourier_pad(0, rank, comm_);
  const zisa::int_t i1_pad = grid_.i_fourier_pad(0, rank + 1, comm_);
  // Send unpadded data to other ranks
  for (zisa::int_t d = 0; d < n_vars; ++d) {
    const auto slice = component(u_hat, d);
    for (int send_to = 0; send_to < size; ++send_to) {
      const zisa::int_t i0_other_pad = grid_.i_fourier_pad(0, send_to, comm_);
      const zisa::int_t i1_other_pad
          = grid_.i_fourier_pad(0, send_to + 1, comm_);
      if (i0 < i0_other_pad && i1 > i0_other_pad) {
        send_reqs.emplace_back();
        MPI_Isend(slice.raw() + (i0_other_pad - i0) * slice.shape(1),
                  (i1 - i0_other_pad) * slice.shape(1),
                  mpi_type<complex_t>(),
                  send_to,
                  d,
                  comm_,
                  &send_reqs.back());
      } else if (i0 >= i0_other_pad && i1 <= i1_other_pad) {
        send_reqs.emplace_back();
        MPI_Isend(slice.raw(),
                  slice.size(),
                  mpi_type<complex_t>(),
                  send_to,
                  d,
                  comm_,
                  &send_reqs.back());
      } else if (i0 < i1_other_pad && i1 > i1_other_pad) {
        send_reqs.emplace_back();
        MPI_Isend(slice.raw(),
                  (i1_other_pad - i0) * slice.shape(1),
                  mpi_type<complex_t>(),
                  send_to,
                  d,
                  comm_,
                  &send_reqs.back());
      }
    }
  }
  // Receive padded data from other ranks
  for (zisa::int_t d = 0; d < n_vars; ++d) {
    const auto slice = component(u_hat_partial_pad_, d);
    for (int recv_from = 0; recv_from < size; ++recv_from) {
      const zisa::int_t i0_other = grid_.i_fourier(0, recv_from, comm_);
      const zisa::int_t i1_other = grid_.i_fourier(0, recv_from + 1, comm_);
      if (i0_other < i0_pad && i1_other > i0_pad) {
        recv_reqs.emplace_back();
        MPI_Irecv(slice.raw(),
                  (i1_other - i0_pad) * slice.shape(1),
                  mpi_type<complex_t>(),
                  recv_from,
                  d,
                  comm_,
                  &recv_reqs.back());
      } else if (i0_other >= i0_pad && i1_other <= i1_pad) {
        recv_reqs.emplace_back();
        MPI_Irecv(slice.raw() + (i0_other - i0_pad) * slice.shape(1),
                  (i1_other - i0_other) * slice.shape(1),
                  mpi_type<complex_t>(),
                  recv_from,
                  d,
                  comm_,
                  &recv_reqs.back());
      } else if (i0_other < i1_pad && i1_other > i1_pad) {
        recv_reqs.emplace_back();
        MPI_Irecv(slice.raw() + (i0_other - i0_pad) * slice.shape(1),
                  (i1_pad - i0_other) * slice.shape(1),
                  mpi_type<complex_t>(),
                  recv_from,
                  d,
                  comm_,
                  &recv_reqs.back());
      }
    }
  }
  // Finish padding the data
  if (i1_pad > grid_.N_fourier) {
    zisa::int_t from = 0;
    if (i0_pad < grid_.N_fourier) {
      from = grid_.N_fourier - i0_pad;
    }
    for (zisa::int_t d = 0; d < u_hat_partial_pad_.shape(0); ++d) {
      for (zisa::int_t i = from; i < u_hat_partial_pad_.shape(1); ++i) {
        for (zisa::int_t j = 0; j < u_hat_partial_pad_.shape(2); ++j) {
          u_hat_partial_pad_(d, i, j) = 0;
        }
      }
    }
  }
  MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE);
  for (zisa::int_t d = 0; d < h_u_hat_pad_.shape(0); ++d) {
    for (zisa::int_t i = 0; i < h_u_hat_pad_.shape(1); ++i) {
      for (zisa::int_t j = 0; j < h_u_hat_pad_.shape(2); ++j) {
        if (j < grid_.N_fourier) {
          h_u_hat_pad_(d, i, j) = u_hat_partial_pad_(d, i, j);
        } else if (j < grid_.N_phys_pad - grid_.N_fourier) {
          h_u_hat_pad_(d, i, j) = 0;
        } else {
          h_u_hat_pad_(d, i, j)
              = u_hat_partial_pad_(d, i, j + grid_.N_phys - grid_.N_phys_pad);
        }
      }
    }
  }
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::pad_u_hat");
}

template <>
void IncompressibleEuler_MPI_Base<2>::unpad_B_hat() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::unpad_B_hat");
  int rank, size;
  MPI_Comm_rank(comm_, &rank);
  MPI_Comm_size(comm_, &size);
  std::vector<MPI_Request> send_reqs;
  std::vector<MPI_Request> recv_reqs;
  const zisa::int_t n_vars = B_hat_.shape(0);
  const zisa::int_t i0 = grid_.i_fourier(0, rank, comm_);
  const zisa::int_t i1 = grid_.i_fourier(0, rank + 1, comm_);
  const zisa::int_t i0_pad = grid_.i_fourier_pad(0, rank, comm_);
  const zisa::int_t i1_pad = grid_.i_fourier_pad(0, rank + 1, comm_);
  // Unpad the local dimension
  for (zisa::int_t d = 0; d < B_hat_partial_pad_.shape(0); ++d) {
    for (zisa::int_t i = 0; i < B_hat_partial_pad_.shape(1); ++i) {
      for (zisa::int_t j = 0; j < B_hat_partial_pad_.shape(2); ++j) {
        if (j < grid_.N_fourier) {
          B_hat_partial_pad_(d, i, j) = h_B_hat_pad_(d, i, j);
        } else {
          B_hat_partial_pad_(d, i, j)
              = h_B_hat_pad_(d, i, j + grid_.N_phys_pad - grid_.N_phys);
        }
      }
    }
  }
  // Send the unpadded data to the other ranks
  for (zisa::int_t d = 0; d < n_vars; ++d) {
    const auto slice = component(B_hat_partial_pad_, d);
    for (int send_to = 0; send_to < size; ++send_to) {
      const zisa::int_t i0_other = grid_.i_fourier(0, send_to, comm_);
      const zisa::int_t i1_other = grid_.i_fourier(0, send_to + 1, comm_);
      if (i0_other < i0_pad && i1_other > i0_pad) {
        send_reqs.emplace_back();
        MPI_Isend(slice.raw(),
                  (i1_other - i0_pad) * slice.shape(1),
                  mpi_type<complex_t>(),
                  send_to,
                  d,
                  comm_,
                  &send_reqs.back());
      } else if (i0_other >= i0_pad && i1_other <= i1_pad) {
        send_reqs.emplace_back();
        MPI_Isend(slice.raw() + (i0_other - i0_pad) * slice.shape(1),
                  (i1_other - i0_other) * slice.shape(1),
                  mpi_type<complex_t>(),
                  send_to,
                  d,
                  comm_,
                  &send_reqs.back());
      } else if (i0_other < i1_pad && i1_other > i1_pad) {
        send_reqs.emplace_back();
        MPI_Isend(slice.raw() + (i0_other - i0_pad) * slice.shape(1),
                  (i1_pad - i0_other) * slice.shape(1),
                  mpi_type<complex_t>(),
                  send_to,
                  d,
                  comm_,
                  &send_reqs.back());
      }
    }
  }
  // Receive the data from the other ranks
  for (zisa::int_t d = 0; d < n_vars; ++d) {
    const auto slice = component(B_hat_, d);
    for (int recv_from = 0; recv_from < size; ++recv_from) {
      const zisa::int_t i0_other_pad = grid_.i_fourier_pad(0, recv_from, comm_);
      const zisa::int_t i1_other_pad
          = grid_.i_fourier_pad(0, recv_from + 1, comm_);
      if (i0 < i0_other_pad && i1 > i0_other_pad) {
        recv_reqs.emplace_back();
        MPI_Irecv(slice.raw() + (i0_other_pad - i0) * slice.shape(1),
                  (i1 - i0_other_pad) * slice.shape(1),
                  mpi_type<complex_t>(),
                  recv_from,
                  d,
                  comm_,
                  &recv_reqs.back());
      } else if (i0 >= i0_other_pad && i1 <= i1_other_pad) {
        recv_reqs.emplace_back();
        MPI_Irecv(slice.raw(),
                  slice.size(),
                  mpi_type<complex_t>(),
                  recv_from,
                  d,
                  comm_,
                  &recv_reqs.back());
      } else if (i0 < i1_other_pad && i1 > i1_other_pad) {
        recv_reqs.emplace_back();
        MPI_Irecv(slice.raw(),
                  (i1_other_pad - i0) * slice.shape(1),
                  mpi_type<complex_t>(),
                  recv_from,
                  d,
                  comm_,
                  &recv_reqs.back());
      }
    }
  }
  MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE);
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::unpad_B_hat");
}

template <>
IncompressibleEuler_MPI_Base<3>::IncompressibleEuler_MPI_Base(
    const Grid<3> &grid, MPI_Comm comm, bool has_tracer)
    : super(grid), comm_(comm), has_tracer_(has_tracer) {
  const zisa::shape_t<4> shape_unpad = grid_.shape_fourier(1, comm);
  const zisa::shape_t<4> shape_pad = grid_.shape_fourier_pad(1, comm);
  const zisa::int_t n_var_u = 3 + (has_tracer ? 1 : 0);
  const zisa::int_t n_var_B = 6 + (has_tracer ? 3 : 0);
  u_hat_partial_pad_ = zisa::array<complex_t, 4>(
      zisa::shape_t<4>(n_var_u, shape_pad[1], shape_unpad[2], shape_unpad[3]),
      zisa::device_type::cpu);
  h_u_hat_pad_
      = grid_.make_array_fourier_pad(n_var_u, zisa::device_type::cpu, comm);
  d_u_hat_pad_
      = grid_.make_array_fourier_pad(n_var_u, zisa::device_type::cuda, comm);
  u_pad_ = grid_.make_array_phys_pad(n_var_u, zisa::device_type::cuda, comm);
  B_pad_ = grid_.make_array_phys_pad(n_var_B, zisa::device_type::cuda, comm);
  d_B_hat_pad_
      = grid_.make_array_fourier_pad(n_var_B, zisa::device_type::cuda, comm);
  h_B_hat_pad_
      = grid_.make_array_fourier_pad(n_var_B, zisa::device_type::cpu, comm);
  B_hat_partial_pad_ = zisa::array<complex_t, 4>(
      zisa::shape_t<4>(n_var_B, shape_pad[1], shape_unpad[2], shape_unpad[3]),
      zisa::device_type::cpu);
  B_hat_ = grid_.make_array_fourier(n_var_B, zisa::device_type::cpu, comm);
  fft_u_ = make_fft_mpi<3>(d_u_hat_pad_, u_pad_, comm, FFT_BACKWARD);
  fft_B_ = make_fft_mpi<3>(d_B_hat_pad_, B_pad_, comm, FFT_FORWARD);
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

template <>
void IncompressibleEuler_MPI_Base<3>::computeB() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::computeB");
  if (has_tracer_) {
    incompressible_euler_compute_B_tracer_cuda<3>(
        fft_B_->u(), fft_u_->u(), grid_);
  } else {
    incompressible_euler_compute_B_cuda<3>(fft_B_->u(), fft_u_->u(), grid_);
  }
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::computeB");
}

template <>
void IncompressibleEuler_MPI_Base<3>::pad_u_hat(
    const zisa::array_const_view<complex_t, 4> &u_hat) {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::pad_u_hat");
  LOG_ERR("Not Implemented");
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::pad_u_hat");
}

template <>
void IncompressibleEuler_MPI_Base<3>::unpad_B_hat() {
  AZEBAN_PROFILE_START("IncompressibleEuler_MPI::unpad_B_hat");
  LOG_ERR("Not Implemented");
  AZEBAN_PROFILE_STOP("IncompressibleEuler_MPI::unpad_B_hat");
}

}

#endif
