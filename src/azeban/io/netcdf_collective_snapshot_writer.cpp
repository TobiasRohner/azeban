#include <azeban/io/netcdf_collective_snapshot_writer.hpp>
#include <azeban/operations/copy_to_padded.hpp>
#include <azeban/profiler.hpp>
#include <filesystem>
#include <netcdf.h>
#include <netcdf_par.h>
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif

namespace azeban {

#define CHECK_NETCDF(...)                                                      \
  if (int status = (__VA_ARGS__); status != NC_NOERR) {                        \
    LOG_ERR(nc_strerror(status));                                              \
  }

static constexpr nc_type netcdf_type(float) { return NC_FLOAT; }
static constexpr nc_type netcdf_type(double) { return NC_DOUBLE; }

template <int Dim>
NetCDFCollectiveSnapshotWriter<Dim>::NetCDFCollectiveSnapshotWriter(
    const std::string &path,
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    bool has_tracer,
    zisa::int_t num_samples,
    zisa::int_t sample_idx_start,
    bool save_pressure,
    void *work_area)
    : super(grid, snapshot_times, sample_idx_start),
      save_pressure_(save_pressure),
      work_area_(work_area) {
#if AZEBAN_HAS_MPI
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size > 1) {
    CHECK_NETCDF(nc_create_par(path.c_str(),
                               NC_CLOBBER | NC_NETCDF4,
                               MPI_COMM_WORLD,
                               MPI_INFO_NULL,
                               &ncid_));
  } else {
    CHECK_NETCDF(nc_create(path.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid_));
  }
#else
  CHECK_NETCDF(nc_create(path.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid_));
#endif
  CHECK_NETCDF(nc_set_fill(ncid_, NC_NOFILL, NULL));
  setup_file(grid, snapshot_times, has_tracer, save_pressure, num_samples);
  if (save_pressure_) {
    u_ = grid.make_array_phys(Dim, zisa::device_type::cpu);
    u_hat_ = grid.make_array_fourier(Dim, zisa::device_type::cpu);
    u_hat_pad_ = grid.make_array_fourier_pad(Dim, zisa::device_type::cpu);
    u_pad_ = grid.make_array_phys_pad(Dim, zisa::device_type::cpu);
    B_pad_ = grid.make_array_phys_pad((Dim * Dim + Dim) / 2,
                                      zisa::device_type::cpu);
    B_hat_pad_ = grid.make_array_fourier_pad((Dim * Dim + Dim) / 2,
                                             zisa::device_type::cpu);
    p_hat_ = grid.make_array_fourier(1, zisa::device_type::cpu);
    p_ = grid.make_array_phys(1, zisa::device_type::cpu);
    fft_u_ = make_fft<Dim>(u_hat_, u_, FFT_FORWARD);
    fft_u_pad_ = make_fft<Dim>(u_hat_pad_, u_pad_, FFT_BACKWARD);
    fft_B_ = make_fft<Dim>(B_hat_pad_, B_pad_, FFT_FORWARD);
    fft_p_ = make_fft<Dim>(p_hat_, p_, FFT_BACKWARD);
  }
}

template <int Dim>
NetCDFCollectiveSnapshotWriter<Dim>::~NetCDFCollectiveSnapshotWriter() {
  CHECK_NETCDF(nc_close(ncid_));
}

template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u, real_t) {
  ProfileHost pofile("NetCDFCollectiveSnapshotWriter::write");
  LOG_ERR_IF(u.memory_location() != zisa::device_type::cpu,
             "u must be on the host");
  size_t slice_size = 1;
  for (int i = 0; i < Dim; ++i) {
    slice_size *= u.shape(i + 1);
  }
  size_t start[Dim + 2];
  size_t count[Dim + 2];
  start[0] = sample_idx_;
  count[0] = 1;
  start[1] = snapshot_idx_;
  count[1] = 1;
  for (int i = 0; i < Dim; ++i) {
    start[i + 2] = 0;
    count[i + 2] = u.shape(i + 1);
  }
  CHECK_NETCDF(
      nc_put_vara(ncid_, varids_[0], start, count, &u[0 * slice_size]));
  CHECK_NETCDF(
      nc_put_vara(ncid_, varids_[1], start, count, &u[1 * slice_size]));
  if (Dim > 2) {
    CHECK_NETCDF(
        nc_put_vara(ncid_, varids_[2], start, count, &u[2 * slice_size]));
  }
  if (u.shape(0) == Dim + 1) {
    CHECK_NETCDF(
        nc_put_vara(ncid_, varids_[3], start, count, &u[Dim * slice_size]));
  }
  if (save_pressure_) {
    zisa::copy(u_, u);
    compute_u_hat();
    compute_u_hat_pad();
    compute_u_pad();
    compute_B_pad();
    compute_B_hat_pad();
    compute_p_hat();
    compute_p();
    CHECK_NETCDF(nc_put_vara(ncid_, varids_[4], start, count, p_.raw()));
  }
  ++snapshot_idx_;
}

template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat, real_t) {}

#if AZEBAN_HAS_MPI
template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u,
    real_t t,
    const Communicator *comm) {
  ProfileHost pofile("NetCDFCollectiveSnapshotWriter::write");
  LOG_ERR_IF(u.memory_location() != zisa::device_type::cpu,
             "u must be on the host");
  const int rank = comm->rank();
  const int size = comm->size();

  std::vector<int> cnts(size);
  std::vector<int> displs(size);
  for (int r = 0; r < size; ++r) {
    cnts[r] = zisa::pow<Dim - 1>(grid_.N_phys)
              * (grid_.N_phys / size
                 + (zisa::integer_cast<zisa::int_t>(r) < grid_.N_phys % size));
  }
  displs[0] = 0;
  for (int r = 1; r < size; ++r) {
    displs[r] = displs[r - 1] + cnts[r - 1];
  }
  const zisa::int_t n_elems_per_component_glob
      = zisa::product(grid_.shape_phys(1));
  const zisa::int_t n_elems_per_component_loc
      = zisa::product(grid_.shape_phys(1, comm));

  std::vector<MPI_Request> reqs(u.shape(0));
  for (zisa::int_t i = 0; i < u.shape(0); ++i) {
    MPI_Igatherv(u.raw() + i * n_elems_per_component_loc,
                 cnts[rank],
                 mpi_type<real_t>(),
                 u_.raw() + i * n_elems_per_component_glob,
                 cnts.data(),
                 displs.data(),
                 mpi_type<real_t>(),
                 0,
                 comm->get_mpi_comm(),
                 &reqs[i]);
  }
  MPI_Waitall(u.shape(0), reqs.data(), MPI_STATUSES_IGNORE);
  if (rank == 0) {
    size_t slice_size = 1;
    for (int i = 0; i < Dim; ++i) {
      slice_size *= u_.shape(i + 1);
    }
    size_t start[Dim + 2];
    size_t count[Dim + 2];
    start[0] = sample_idx_;
    count[0] = 1;
    start[1] = snapshot_idx_;
    count[1] = 1;
    for (int i = 0; i < Dim; ++i) {
      start[i + 2] = 0;
      count[i + 2] = u_.shape(i + 1);
    }
    CHECK_NETCDF(
        nc_put_vara(ncid_, varids_[0], start, count, &u_[0 * slice_size]));
    CHECK_NETCDF(
        nc_put_vara(ncid_, varids_[1], start, count, &u_[1 * slice_size]));
    if (Dim > 2) {
      CHECK_NETCDF(
          nc_put_vara(ncid_, varids_[2], start, count, &u_[2 * slice_size]));
    }
    if (u.shape(0) == Dim + 1) {
      CHECK_NETCDF(
          nc_put_vara(ncid_, varids_[3], start, count, &u_[Dim * slice_size]));
    }
    if (save_pressure_) {
      compute_u_hat();
      compute_u_hat_pad();
      compute_u_pad();
      compute_B_pad();
      compute_B_hat_pad();
      compute_p_hat();
      compute_p();
      CHECK_NETCDF(nc_put_vara(ncid_, varids_[4], start, count, p_.raw()));
    }
  }
  ++snapshot_idx_;
}

template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &,
    real_t,
    const Communicator *) {
  LOG_ERR("Not yet implemented");
}
#endif

template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::setup_file(
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    bool has_tracer,
    bool save_pressure,
    zisa::int_t num_samples) {
  CHECK_NETCDF(nc_def_dim(ncid_, "member", num_samples, &dimids_[0]));
  CHECK_NETCDF(nc_def_dim(ncid_, "time", snapshot_times.size(), &dimids_[1]));
  CHECK_NETCDF(nc_def_dim(ncid_, "x", grid.N_phys, &dimids_[2]));
  CHECK_NETCDF(nc_def_dim(ncid_, "y", grid.N_phys, &dimids_[3]));
  if (Dim > 2) {
    CHECK_NETCDF(nc_def_dim(ncid_, "z", grid.N_phys, &dimids_[4]));
  }

  int varid_member, varid_time, varid_x, varid_y, varid_z;
  CHECK_NETCDF(
      nc_def_var(ncid_, "member", NC_INT, 1, &dimids_[0], &varid_member));
  CHECK_NETCDF(nc_def_var(
      ncid_, "time", netcdf_type(real_t{}), 1, &dimids_[1], &varid_time));
  CHECK_NETCDF(
      nc_def_var(ncid_, "x", netcdf_type(real_t{}), 1, &dimids_[2], &varid_x));
  CHECK_NETCDF(
      nc_def_var(ncid_, "y", netcdf_type(real_t{}), 1, &dimids_[3], &varid_y));
  if (Dim > 2) {
    CHECK_NETCDF(nc_def_var(
        ncid_, "z", netcdf_type(real_t{}), 1, &dimids_[4], &varid_z));
  }
  size_t chunksizes[Dim + 2];
  chunksizes[0] = 1;
  chunksizes[1] = 1;
  for (int i = 0; i < Dim; ++i) {
    chunksizes[i + 2] = grid.N_phys;
  }
  CHECK_NETCDF(nc_def_var(
      ncid_, "u", netcdf_type(real_t{}), 2 + Dim, dimids_, &varids_[0]));
  CHECK_NETCDF(nc_def_var_chunking(ncid_, varids_[0], NC_CHUNKED, chunksizes));
  CHECK_NETCDF(nc_def_var(
      ncid_, "v", netcdf_type(real_t{}), 2 + Dim, dimids_, &varids_[1]));
  CHECK_NETCDF(nc_def_var_chunking(ncid_, varids_[1], NC_CHUNKED, chunksizes));
  if (Dim > 2) {
    CHECK_NETCDF(nc_def_var(
        ncid_, "w", netcdf_type(real_t{}), 2 + Dim, dimids_, &varids_[2]));
    CHECK_NETCDF(
        nc_def_var_chunking(ncid_, varids_[2], NC_CHUNKED, chunksizes));
  }
  if (has_tracer) {
    CHECK_NETCDF(nc_def_var(
        ncid_, "rho", netcdf_type(real_t{}), 2 + Dim, dimids_, &varids_[3]));
    CHECK_NETCDF(
        nc_def_var_chunking(ncid_, varids_[3], NC_CHUNKED, chunksizes));
  }
  if (save_pressure) {
    CHECK_NETCDF(nc_def_var(
        ncid_, "p", netcdf_type(real_t{}), 2 + Dim, dimids_, &varids_[4]));
    CHECK_NETCDF(
        nc_def_var_chunking(ncid_, varids_[4], NC_CHUNKED, chunksizes));
  }

  std::vector<int> member(num_samples);
  for (zisa::int_t i = 0; i < num_samples; ++i) {
    member[i] = i;
  }
  CHECK_NETCDF(nc_put_var(ncid_, varid_member, member.data()));
  CHECK_NETCDF(nc_put_var(ncid_, varid_time, snapshot_times.data()));
  std::vector<real_t> space(grid.N_phys);
  for (zisa::int_t i = 0; i < grid.N_phys; ++i) {
    const real_t x = static_cast<real_t>(i) / grid.N_phys;
    space[i] = x;
  }
  CHECK_NETCDF(nc_put_var(ncid_, varid_x, space.data()));
  CHECK_NETCDF(nc_put_var(ncid_, varid_y, space.data()));
  if (Dim > 2) {
    CHECK_NETCDF(nc_put_var(ncid_, varid_z, space.data()));
  }
}

template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::compute_u_hat() {
  fft_u_->forward();
}

template <>
void NetCDFCollectiveSnapshotWriter<1>::compute_u_hat_pad() {}

template <>
void NetCDFCollectiveSnapshotWriter<2>::compute_u_hat_pad() {
  zisa::shape_t<2> slice_u_hat_shape(u_hat_.shape(1), u_hat_.shape(2));
  zisa::shape_t<2> slice_u_hat_pad_shape(u_hat_pad_.shape(1),
                                         u_hat_pad_.shape(2));
  for (int i = 0; i < 2; ++i) {
    const zisa::array_const_view<complex_t, 2> slice_u_hat(
        slice_u_hat_shape,
        u_hat_.raw() + i * zisa::product(slice_u_hat_shape),
        zisa::device_type::cpu);
    const zisa::array_view<complex_t, 2> slice_u_hat_pad(
        slice_u_hat_pad_shape,
        u_hat_pad_.raw() + i * zisa::product(slice_u_hat_pad_shape),
        zisa::device_type::cpu);
    copy_to_padded(slice_u_hat_pad, slice_u_hat);
  }
}

template <>
void NetCDFCollectiveSnapshotWriter<3>::compute_u_hat_pad() {
  zisa::shape_t<3> slice_u_hat_shape(
      u_hat_.shape(1), u_hat_.shape(2), u_hat_.shape(3));
  zisa::shape_t<3> slice_u_hat_pad_shape(
      u_hat_pad_.shape(1), u_hat_pad_.shape(2), u_hat_pad_.shape(3));
  for (int i = 0; i < 3; ++i) {
    const zisa::array_const_view<complex_t, 3> slice_u_hat(
        slice_u_hat_shape,
        u_hat_.raw() + i * zisa::product(slice_u_hat_shape),
        zisa::device_type::cpu);
    const zisa::array_view<complex_t, 3> slice_u_hat_pad(
        slice_u_hat_pad_shape,
        u_hat_pad_.raw() + i * zisa::product(slice_u_hat_pad_shape),
        zisa::device_type::cpu);
    copy_to_padded(slice_u_hat_pad, slice_u_hat);
  }
}

template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::compute_u_pad() {
  fft_u_->backward();
}

template <>
void NetCDFCollectiveSnapshotWriter<1>::compute_B_pad() {}

template <>
void NetCDFCollectiveSnapshotWriter<2>::compute_B_pad() {
  const zisa::int_t N_phys = p_.shape(1);
  const zisa::int_t N_phys_pad = u_pad_.shape(1);
  const real_t norm = 1.0 / (N_phys * N_phys_pad);
#pragma omp parallel for collapse(2)
  for (zisa::int_t i = 0; i < u_pad_.shape(1); ++i) {
    for (zisa::int_t j = 0; j < u_pad_.shape(2); ++j) {
      const real_t u1 = norm * u_pad_(0, i, j);
      const real_t u2 = norm * u_pad_(1, i, j);
      B_pad_(0, i, j) = u1 * u1;
      B_pad_(1, i, j) = u2 * u1;
      B_pad_(2, i, j) = u2 * u2;
    }
  }
}

template <>
void NetCDFCollectiveSnapshotWriter<3>::compute_B_pad() {
  const zisa::int_t N_phys = p_.shape(1);
  const zisa::int_t N_phys_pad = u_pad_.shape(1);
  const real_t norm = 1.0
                      / (zisa::pow<3>(zisa::sqrt(N_phys))
                         * zisa::pow<3>(zisa::sqrt(N_phys_pad)));
#pragma omp parallel for collapse(3)
  for (zisa::int_t i = 0; i < u_pad_.shape(1); ++i) {
    for (zisa::int_t j = 0; j < u_pad_.shape(2); ++j) {
      for (zisa::int_t k = 0; k < u_pad_.shape(3); ++k) {
        const real_t u1 = norm * u_pad_(0, i, j, k);
        const real_t u2 = norm * u_pad_(1, i, j, k);
        const real_t u3 = norm * u_pad_(2, i, j, k);
        B_pad_(0, i, j, k) = u1 * u1;
        B_pad_(1, i, j, k) = u2 * u1;
        B_pad_(2, i, j, k) = u2 * u2;
        B_pad_(3, i, j, k) = u3 * u1;
        B_pad_(4, i, j, k) = u3 * u2;
        B_pad_(5, i, j, k) = u3 * u3;
      }
    }
  }
}

template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::compute_B_hat_pad() {
  fft_B_->forward();
}

template <>
void NetCDFCollectiveSnapshotWriter<1>::compute_p_hat() {}

template <>
void NetCDFCollectiveSnapshotWriter<2>::compute_p_hat() {
  const long N_phys = p_.shape(1);
  const long N_fourier = N_phys / 2 + 1;
  const long N_phys_pad = u_pad_.shape(1);
  const real_t norm = 1.0 / zisa::pow<2>(N_phys);
#pragma omp parallel for collapse(2)
  for (long i = 0; i < N_phys; ++i) {
    for (long j = 0; j < N_fourier; ++j) {
      const long i_B = i >= N_fourier ? N_phys_pad - N_phys + i : i;
      const long i_ = i >= N_fourier ? i - N_phys : i;
      const real_t k1 = 2 * zisa::pi * i_;
      const real_t k2 = 2 * zisa::pi * j;
      const real_t absk2 = k1 * k1 + k2 * k2;
      const complex_t B11 = B_hat_pad_(0, i_B, j);
      const complex_t B12 = B_hat_pad_(1, i_B, j);
      const complex_t B21 = B12;
      const complex_t B22 = B_hat_pad_(2, i_B, j);
      p_hat_(0, i, j)
          = -norm
            * (k1 * k1 * B11 + k1 * k2 * B12 + k2 * k1 * B21 + k2 * k2 * B22)
            / absk2;
    }
  }
  p_hat_(0, 0, 0) = 0;
}

template <>
void NetCDFCollectiveSnapshotWriter<3>::compute_p_hat() {
  const long N_phys = p_.shape(1);
  const long N_fourier = N_phys / 2 + 1;
  const long N_phys_pad = u_pad_.shape(1);
  const real_t norm = 1.0 / zisa::pow<3>(N_phys);
#pragma omp parallel for collapse(3)
  for (long i = 0; i < N_phys; ++i) {
    for (long j = 0; j < N_phys; ++j) {
      for (long k = 0; k < N_fourier; ++k) {
        const long i_B = i >= N_fourier ? N_phys_pad - N_phys + i : i;
        const long j_B = j >= N_fourier ? N_phys_pad - N_phys + j : j;
        const long i_ = i >= N_fourier ? i - N_phys : i;
        const long j_ = j >= N_fourier ? j - N_phys : j;
        const real_t k1 = 2 * zisa::pi * i_;
        const real_t k2 = 2 * zisa::pi * j_;
        const real_t k3 = 2 * zisa::pi * k;
        const real_t absk2 = k1 * k1 + k2 * k2 + k3 * k3;
        const complex_t B11 = B_hat_pad_(0, i_B, j_B, k);
        const complex_t B12 = B_hat_pad_(1, i_B, j_B, k);
        const complex_t B13 = B_hat_pad_(3, i_B, j_B, k);
        const complex_t B21 = B12;
        const complex_t B22 = B_hat_pad_(2, i_B, j_B, k);
        const complex_t B23 = B_hat_pad_(4, i_B, j_B, k);
        const complex_t B31 = B13;
        const complex_t B32 = B23;
        const complex_t B33 = B_hat_pad_(5, i_B, j_B, k);
        p_hat_(0, i, j, k) = -norm
                             * (k1 * k1 * B11 + k1 * k2 * B12 + k1 * k3 * B13
                                + k2 * k1 * B21 + k2 * k2 * B22 + k2 * k3 * B23
                                + k3 * k1 * B31 + k3 * k2 * B32 + k3 * k3 * B33)
                             / absk2;
      }
    }
  }
  p_hat_(0, 0, 0, 0) = 0;
}

template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::compute_p() {
  fft_p_->backward();
}

template class NetCDFCollectiveSnapshotWriter<1>;
template class NetCDFCollectiveSnapshotWriter<2>;
template class NetCDFCollectiveSnapshotWriter<3>;

}
