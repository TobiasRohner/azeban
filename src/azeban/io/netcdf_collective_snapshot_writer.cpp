#include <azeban/io/netcdf_collective_snapshot_writer.hpp>
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
    void *work_area)
    : super(grid, snapshot_times, sample_idx_start), work_area_(work_area) {
#if AZEBAN_HAS_MPI
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size > 1) {
    CHECK_NETCDF(nc_create_par(
        path.c_str(), NC_NETCDF4, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid_));
  } else {
    CHECK_NETCDF(nc_create(path.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid_));
  }
#else
  CHECK_NETCDF(nc_create(path.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid_));
#endif
  CHECK_NETCDF(nc_set_fill(ncid_, NC_NOFILL, NULL));
  setup_file(grid, snapshot_times, has_tracer, num_samples);
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
  ++snapshot_idx_;
}

template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &, real_t) {}

#if AZEBAN_HAS_MPI
template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &,
    real_t,
    const Communicator *) {
  LOG_ERR("Not yet implemented");
}

template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &,
    real_t,
    const Communicator *) {}
#endif

template <int Dim>
void NetCDFCollectiveSnapshotWriter<Dim>::setup_file(
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    bool has_tracer,
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
        nc_def_var_chunking(ncid_, varids_[2], NC_CHUNKED, chunksizes));
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

template class NetCDFCollectiveSnapshotWriter<1>;
template class NetCDFCollectiveSnapshotWriter<2>;
template class NetCDFCollectiveSnapshotWriter<3>;

}
