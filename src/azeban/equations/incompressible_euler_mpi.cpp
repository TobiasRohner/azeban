#if AZEBAN_HAS_MPI

#include <azeban/equations/incompressible_euler_mpi.hpp>

namespace azeban {

template <>
IncompressibleEuler_MPI_Base<2>::IncompressibleEuler_MPI_Base(
    const Grid<2> &grid, MPI_Comm comm, bool has_tracer)
    : super(grid), comm_(comm), has_tracer_(has_tracer) {
  // TODO: Actually pad the padded arrays
  grid_.N_phys_pad = grid_.N_phys;
  grid_.N_fourier_pad = grid_.N_fourier;
  const zisa::int_t n_var_u = 2 + (has_tracer ? 1 : 0);
  const zisa::int_t n_var_B = 3 + (has_tracer ? 2 : 0);
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
IncompressibleEuler_MPI_Base<3>::IncompressibleEuler_MPI_Base(
    const Grid<3> &grid, MPI_Comm comm, bool has_tracer)
    : super(grid), comm_(comm), has_tracer_(has_tracer) {
  // TODO: Actually pad the padded arrays
  grid_.N_phys_pad = grid_.N_phys;
  grid_.N_fourier_pad = grid_.N_fourier;
  const zisa::int_t n_var_u = 3 + (has_tracer ? 1 : 0);
  const zisa::int_t n_var_B = 6 + (has_tracer ? 3 : 0);
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

}

#endif
