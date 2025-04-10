#include <azeban/io/netcdf_enstrophy_spectrum_writer.hpp>
#include <azeban/netcdf.hpp>
#include <azeban/operations/enstrophy_spectrum.hpp>

namespace azeban {

template <int Dim>
NetCDFEnstrophySpectrumWriter<Dim>::NetCDFEnstrophySpectrumWriter(
    int ncid,
    const Grid<Dim> &grid,
    const std::vector<double> &snapshot_times,
    int sample_idx_start)
    : super(ncid, grid, snapshot_times, sample_idx_start) {
  // Define own group
  CHECK_NETCDF(nc_def_grp(ncid_, "enstrophy_spectrum", &grpid_));
  // Define dimensions
  int dimid_member;
  int dimid_time;
  int dimid_k;
  CHECK_NETCDF(nc_inq_dimid(ncid_, "member", &dimid_member));
  CHECK_NETCDF(nc_def_dim(grpid_, "time", snapshot_times.size(), &dimid_time));
  CHECK_NETCDF(nc_def_dim(grpid_, "k", grid.N_fourier, &dimid_k));
  // Define variables
  int varid_sim_time;
  int varid_k;
  CHECK_NETCDF(
      nc_def_var(grpid_, "time", NC_DOUBLE, 1, &dimid_time, &varid_sim_time));
  CHECK_NETCDF(nc_def_var(grpid_, "k", NC_REAL, 1, &dimid_k, &varid_k));
  const int dimids_real_time[2] = {dimid_member, dimid_time};
  CHECK_NETCDF(nc_def_var(
      grpid_, "real_time", NC_DOUBLE, 2, dimids_real_time, &varid_real_time_));
  CHECK_NETCDF(
      nc_put_att_text(grpid_, varid_real_time_, "units", 8, "seconds"));
  const int dimids_Ek[3] = {dimid_member, dimid_time, dimid_k};
  const size_t Ek_chunksizes[3] = {1, 1, grid.N_fourier};
  CHECK_NETCDF(nc_def_var(grpid_, "Ek", NC_REAL, 3, dimids_Ek, &varid_ek_));
  CHECK_NETCDF(
      nc_def_var_chunking(grpid_, varid_ek_, NC_CHUNKED, Ek_chunksizes));
  // Initialize variables
  CHECK_NETCDF(nc_put_var(grpid_, varid_sim_time, snapshot_times.data()));
  std::vector<real_t> k(grid.N_fourier);
  for (zisa::int_t i = 0; i < grid.N_fourier; ++i) {
    k[i] = 2 * zisa::pi * i;
  }
  CHECK_NETCDF(nc_put_var(grpid_, varid_k, k.data()));
  // Record the current time
  start_time_ = std::chrono::steady_clock::now();
}

template <int Dim>
void NetCDFEnstrophySpectrumWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &, double) {
  // Nothing to do here
}

template <int Dim>
void NetCDFEnstrophySpectrumWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat, double) {
  const std::vector<real_t> spectrum = enstrophy_spectrum(grid_, u_hat);
  const size_t start[3] = {sample_idx_, snapshot_idx_, 0};
  const size_t count[3] = {1, 1, grid_.N_fourier};
  CHECK_NETCDF(nc_put_vara(grpid_, varid_ek_, start, count, spectrum.data()));
  const auto time = std::chrono::steady_clock::now();
  const auto elapsed
      = std::chrono::duration_cast<std::chrono::duration<double>>(
          time - start_time_);
  const size_t index[2] = {sample_idx_, snapshot_idx_};
  const double elapsed_count = elapsed.count();
  CHECK_NETCDF(nc_put_var1(grpid_, varid_real_time_, index, &elapsed_count));
  ++snapshot_idx_;
}

#if AZEBAN_HAS_MPI
template <int Dim>
void NetCDFEnstrophySpectrumWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &,
    double,
    const Communicator *) {
  // Nothing to do here
}

template <int Dim>
void NetCDFEnstrophySpectrumWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    double,
    const Communicator *comm) {
  const std::vector<real_t> spectrum
      = enstrophy_spectrum(grid_, u_hat, comm->get_mpi_comm());
  if (comm->rank() == 0) {
    const size_t start[3] = {sample_idx_, snapshot_idx_, 0};
    const size_t count[3] = {1, 1, grid_.N_fourier};
    CHECK_NETCDF(nc_put_vara(grpid_, varid_ek_, start, count, spectrum.data()));
    const auto time = std::chrono::steady_clock::now();
    const auto elapsed
        = std::chrono::duration_cast<std::chrono::duration<double>>(
            time - start_time_);
    const size_t index[2] = {sample_idx_, snapshot_idx_};
    const double elapsed_count = elapsed.count();
    CHECK_NETCDF(nc_put_var1(grpid_, varid_real_time_, index, &elapsed_count));
  }
  ++snapshot_idx_;
}
#endif

template class NetCDFEnstrophySpectrumWriter<1>;
template class NetCDFEnstrophySpectrumWriter<2>;
template class NetCDFEnstrophySpectrumWriter<3>;

}
