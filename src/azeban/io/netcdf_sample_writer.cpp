#include <azeban/io/netcdf_sample_writer.hpp>
#include <azeban/netcdf.hpp>
#include <azeban/operations/copy_from_padded.hpp>
#include <azeban/operations/scale.hpp>

namespace azeban {

template <int Dim>
NetCDFSampleWriter<Dim>::NetCDFSampleWriter(
    int ncid,
    const Grid<Dim> &grid,
    zisa::int_t N,
    const std::vector<real_t> &snapshot_times,
    bool has_tracer,
    zisa::int_t sample_idx_start)
    : super(ncid, grid, snapshot_times, sample_idx_start),
      N_(N),
      has_tracer_(has_tracer) {
  // Define own group
  CHECK_NETCDF(
      nc_def_grp(ncid_, ("flow_field_" + std::to_string(N_)).c_str(), &grpid_));
  // Define dimensions
  int dimid_member;
  int dimid_time;
  int dimid_dims[3];
  CHECK_NETCDF(nc_inq_dimid(ncid_, "member", &dimid_member));
  CHECK_NETCDF(nc_def_dim(grpid_, "time", snapshot_times.size(), &dimid_time));
  if (N == grid_.N_phys) {
    CHECK_NETCDF(nc_inq_dimid(ncid_, "x", dimid_dims));
    if constexpr (Dim > 1) {
      CHECK_NETCDF(nc_inq_dimid(ncid_, "y", dimid_dims + 1));
    }
    if constexpr (Dim > 2) {
      CHECK_NETCDF(nc_inq_dimid(ncid_, "z", dimid_dims + 2));
    }
  } else {
    CHECK_NETCDF(nc_def_dim(grpid_, "x", N, dimid_dims));
    if constexpr (Dim > 1) {
      CHECK_NETCDF(nc_def_dim(grpid_, "y", N, dimid_dims + 1));
    }
    if constexpr (Dim > 2) {
      CHECK_NETCDF(nc_def_dim(grpid_, "z", N, dimid_dims + 2));
    }
  }
  // Define variables
  int varid_sim_time;
  int varids_xyz[3];
  CHECK_NETCDF(
      nc_def_var(grpid_, "time", NC_REAL, 1, &dimid_time, &varid_sim_time));
  if (N != grid_.N_phys) {
    CHECK_NETCDF(nc_def_var(grpid_, "x", NC_REAL, 1, dimid_dims, varids_xyz));
    if constexpr (Dim > 1) {
      CHECK_NETCDF(
          nc_def_var(grpid_, "y", NC_REAL, 1, dimid_dims + 1, varids_xyz + 1));
    }
    if constexpr (Dim > 2) {
      CHECK_NETCDF(
          nc_def_var(grpid_, "z", NC_REAL, 1, dimid_dims + 2, varids_xyz + 2));
    }
  }
  const int dimids_real_time[2] = {dimid_member, dimid_time};
  CHECK_NETCDF(nc_def_var(
      grpid_, "real_time", NC_REAL, 2, dimids_real_time, &varid_real_time_));
  CHECK_NETCDF(
      nc_put_att_text(grpid_, varid_real_time_, "units", 8, "seconds"));
  const int dimids_fields[5]
      = {dimid_member, dimid_time, dimid_dims[0], dimid_dims[1], dimid_dims[2]};
  const size_t fields_chunksizes[5] = {1, 1, N, N, N};
  CHECK_NETCDF(
      nc_def_var(grpid_, "u", NC_REAL, 2 + Dim, dimids_fields, varids_uvw_));
  CHECK_NETCDF(nc_def_var_chunking(
      grpid_, varids_uvw_[0], NC_CHUNKED, fields_chunksizes));
  if constexpr (Dim > 1) {
    CHECK_NETCDF(nc_def_var(
        grpid_, "v", NC_REAL, 2 + Dim, dimids_fields, varids_uvw_ + 1));
    CHECK_NETCDF(nc_def_var_chunking(
        grpid_, varids_uvw_[1], NC_CHUNKED, fields_chunksizes));
  }
  if constexpr (Dim > 2) {
    CHECK_NETCDF(nc_def_var(
        grpid_, "w", NC_REAL, 2 + Dim, dimids_fields, varids_uvw_ + 2));
    CHECK_NETCDF(nc_def_var_chunking(
        grpid_, varids_uvw_[2], NC_CHUNKED, fields_chunksizes));
  }
  if (has_tracer_) {
    CHECK_NETCDF(nc_def_var(
        grpid_, "rho", NC_REAL, 2 + Dim, dimids_fields, varids_uvw_ + Dim));
    CHECK_NETCDF(nc_def_var_chunking(
        grpid_, varids_uvw_[Dim], NC_CHUNKED, fields_chunksizes));
  }
  // Initialize variables
  CHECK_NETCDF(nc_put_var(grpid_, varid_sim_time, snapshot_times.data()));
  if (N != grid_.N_phys) {
    std::vector<real_t> x(N);
    for (zisa::int_t i = 0; i < N; ++i) {
      x[i] = static_cast<real_t>(i) / N;
    }
    CHECK_NETCDF(nc_put_var(grpid_, varids_xyz[0], x.data()));
    if constexpr (Dim > 1) {
      CHECK_NETCDF(nc_put_var(grpid_, varids_xyz[1], x.data()));
    }
    if constexpr (Dim > 2) {
      CHECK_NETCDF(nc_put_var(grpid_, varids_xyz[2], x.data()));
    }
  }
  // Initialize stuff needed for downsampling
  if (N_ != grid_.N_phys) {
    Grid<Dim> grid_down(N_, grid_.N_phys);
    u_hat_down_ = grid_down.make_array_fourier(Dim + has_tracer_,
                                               zisa::device_type::cpu);
    u_down_
        = grid_down.make_array_phys(Dim + has_tracer_, zisa::device_type::cpu);
    fft_down_ = make_fft<Dim>(u_hat_down_, u_down_, FFT_BACKWARD);
  }
  // Record the current time
  start_time_ = std::chrono::steady_clock::now();
}

template <int Dim>
void NetCDFSampleWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u, real_t) {
  // Store the flow field
  if (N_ == grid_.N_phys) {
    store_u(u);
  }
  // Store the current time
  const auto time = std::chrono::steady_clock::now();
  const auto elapsed
      = std::chrono::duration_cast<std::chrono::duration<real_t>>(
          time - start_time_);
  const size_t index[2] = {sample_idx_, snapshot_idx_};
  const real_t elapsed_count = elapsed.count();
  CHECK_NETCDF(nc_put_var1(grpid_, varid_real_time_, index, &elapsed_count));
  ++snapshot_idx_;
}

template <int Dim>
void NetCDFSampleWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat, real_t) {
  // Store the downsampled flow field
  if (N_ < grid_.N_phys) {
    zisa::shape_t<Dim> slice_shape;
    zisa::shape_t<Dim> slice_shape_down;
    for (int d = 0; d < u_hat.shape(0); ++d) {
      slice_shape[d] = u_hat.shape(d + 1);
      slice_shape_down[d] = u_hat_down_.shape(d + 1);
    }
    for (int i = 0; i < u_hat.shape(0); ++i) {
      zisa::array_const_view<complex_t, Dim> slice(
          slice_shape,
          u_hat.raw() + i * zisa::product(slice_shape),
          u_hat.memory_location());
      zisa::array_view<complex_t, Dim> slice_down(
          slice_shape_down,
          u_hat_down_.raw() + i * zisa::product(slice_shape_down),
          u_hat_down_.device());
      copy_from_padded(slice_down, slice);
    }
    fft_down_->backward();
    scale(real_t{1} / zisa::pow<Dim>(grid_.N_phys), u_down_.view());
    store_u(u_down_);
  }
}

#if AZEBAN_HAS_MPI
template <int Dim>
void NetCDFSampleWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u,
    real_t t,
    const Communicator *comm) {
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

  zisa::array<real_t, Dim + 1> u_full;
  if (rank == 0) {
    u_full = grid_.make_array_phys(u.shape(0), zisa::device_type::cpu);
  }
  std::vector<MPI_Request> reqs(u.shape(0));
  for (zisa::int_t i = 0; i < u.shape(0); ++i) {
    MPI_Igatherv(u.raw() + i * n_elems_per_component_loc,
                 cnts[rank],
                 mpi_type<real_t>(),
                 u_full.raw() + i * n_elems_per_component_glob,
                 cnts.data(),
                 displs.data(),
                 mpi_type<real_t>(),
                 0,
                 comm->get_mpi_comm(),
                 &reqs[i]);
  }
  MPI_Waitall(u.shape(0), reqs.data(), MPI_STATUSES_IGNORE);

  if (rank == 0) {
    write(u_full, t);
  } else {
    ++snapshot_idx_;
  }
}

template <int Dim>
void NetCDFSampleWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    real_t,
    const Communicator *comm) {
  // TODO: Implement
}
#endif

template <int Dim>
void NetCDFSampleWriter<Dim>::store_u(
    const zisa::array_const_view<real_t, Dim + 1> &u) {
  const size_t start[5] = {sample_idx_, snapshot_idx_, 0, 0, 0};
  const size_t count[5] = {1, 1, N_, N_, N_};
  CHECK_NETCDF(nc_put_vara(
      grpid_, varids_uvw_[0], start, count, u.raw() + 0 * zisa::pow<Dim>(N_)));
  if constexpr (Dim > 1) {
    CHECK_NETCDF(nc_put_vara(grpid_,
                             varids_uvw_[1],
                             start,
                             count,
                             u.raw() + 1 * zisa::pow<Dim>(N_)));
  }
  if constexpr (Dim > 2) {
    CHECK_NETCDF(nc_put_vara(grpid_,
                             varids_uvw_[2],
                             start,
                             count,
                             u.raw() + 2 * zisa::pow<Dim>(N_)));
  }
  if (has_tracer_) {
    CHECK_NETCDF(nc_put_vara(grpid_,
                             varids_uvw_[Dim],
                             start,
                             count,
                             u.raw() + Dim * zisa::pow<Dim>(N_)));
  }
}

template class NetCDFSampleWriter<1>;
template class NetCDFSampleWriter<2>;
template class NetCDFSampleWriter<3>;

}
