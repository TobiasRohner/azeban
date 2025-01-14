#include <azeban/io/netcdf_second_order_structure_function_writer.hpp>
#include <azeban/netcdf.hpp>
#include <azeban/operations/second_order_structure_function.hpp>

namespace azeban {

template <int Dim, typename SF>
NetCDFSecondOrderStructureFunctionWriter<Dim, SF>::NetCDFSecondOrderStructureFunctionWriter(
    int ncid,
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    int sample_idx_start)
    : super(ncid, grid, snapshot_times, sample_idx_start) {
  // Define own group
  CHECK_NETCDF(nc_def_grp(ncid_, "second_order_structure_function", &grpid_));
  // Define dimensions
  int dimid_member;
  int dimid_time;
  int dimid_r;
  CHECK_NETCDF(nc_inq_dimid(ncid_, "member", &dimid_member));
  CHECK_NETCDF(nc_def_dim(grpid_, "time", snapshot_times.size(), &dimid_time));
  CHECK_NETCDF(nc_def_dim(grpid_, "r", (grid.N_phys+1)/2, &dimid_r));
  // Define variables
  int varid_sim_time;
  int varid_r;
  CHECK_NETCDF(
      nc_def_var(grpid_, "time", NC_REAL, 1, &dimid_time, &varid_sim_time));
  CHECK_NETCDF(nc_def_var(grpid_, "r", NC_REAL, 1, &dimid_r, &varid_r));
  const int dimids_real_time[2] = {dimid_member, dimid_time};
  CHECK_NETCDF(nc_def_var(
      grpid_, "real_time", NC_REAL, 2, dimids_real_time, &varid_real_time_));
  CHECK_NETCDF(
      nc_put_att_text(grpid_, varid_real_time_, "units", 8, "seconds"));
  const int dimids_S2[3] = {dimid_member, dimid_time, dimid_r};
  const size_t S2_chunksizes[3] = {1, 1, (grid.N_phys+1)/2};
  CHECK_NETCDF(nc_def_var(grpid_, "S2", NC_REAL, 3, dimids_S2, &varid_S2_));
  CHECK_NETCDF(
      nc_def_var_chunking(grpid_, varid_S2_, NC_CHUNKED, S2_chunksizes));
  // Initialize variables
  CHECK_NETCDF(nc_put_var(grpid_, varid_sim_time, snapshot_times.data()));
  std::vector<real_t> r(grid.N_phys/2);
  for (zisa::int_t i = 0; i < (grid.N_phys+1)/2; ++i) {
    r[i] = static_cast<real_t>(i) / grid.N_phys;
  }
  CHECK_NETCDF(nc_put_var(grpid_, varid_r, r.data()));
  // Record the current time
  start_time_ = std::chrono::steady_clock::now();
}

template <int Dim, typename SF>
void NetCDFSecondOrderStructureFunctionWriter<Dim, SF>::write(
    const zisa::array_const_view<real_t, Dim + 1> &, real_t) {
  // Nothing to do here
}

template <int Dim, typename SF>
void NetCDFSecondOrderStructureFunctionWriter<Dim, SF>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat, real_t) {
  const std::vector<real_t> sf = SF::eval(grid_, u_hat);
  const size_t start[3] = {sample_idx_, snapshot_idx_, 0};
  const size_t count[3] = {1, 1, (grid_.N_phys+1)/2};
  CHECK_NETCDF(nc_put_vara(grpid_, varid_S2_, start, count, sf.data()));
  const auto time = std::chrono::steady_clock::now();
  const auto elapsed
      = std::chrono::duration_cast<std::chrono::duration<real_t>>(
          time - start_time_);
  const size_t index[2] = {sample_idx_, snapshot_idx_};
  const real_t elapsed_count = elapsed.count();
  CHECK_NETCDF(nc_put_var1(grpid_, varid_real_time_, index, &elapsed_count));
  ++snapshot_idx_;
}

#if AZEBAN_HAS_MPI
template <int Dim, typename SF>
void NetCDFSecondOrderStructureFunctionWriter<Dim, SF>::write(
    const zisa::array_const_view<real_t, Dim + 1> &,
    real_t,
    const Communicator *) {
  // Nothing to do here
}

template <int Dim, typename SF>
void NetCDFSecondOrderStructureFunctionWriter<Dim, SF>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    real_t,
    const Communicator *comm) {
  const std::vector<real_t> sf = SF::eval(grid_, u_hat, comm->get_mpi_comm());
  if (comm->rank() == 0) {
    const size_t start[3] = {sample_idx_, snapshot_idx_, 0};
    const size_t count[3] = {1, 1, (grid_.N_phys+1)/2};
    CHECK_NETCDF(nc_put_vara(grpid_, varid_S2_, start, count, sf.data()));
    const auto time = std::chrono::steady_clock::now();
    const auto elapsed
        = std::chrono::duration_cast<std::chrono::duration<real_t>>(
            time - start_time_);
    const size_t index[2] = {sample_idx_, snapshot_idx_};
    const real_t elapsed_count = elapsed.count();
    CHECK_NETCDF(nc_put_var1(grpid_, varid_real_time_, index, &elapsed_count));
  }
  ++snapshot_idx_;
}
#endif

template class NetCDFSecondOrderStructureFunctionWriter<1, detail::SF2Exact>;
template class NetCDFSecondOrderStructureFunctionWriter<2, detail::SF2Exact>;
template class NetCDFSecondOrderStructureFunctionWriter<3, detail::SF2Exact>;
template class NetCDFSecondOrderStructureFunctionWriter<1, detail::SF2Approx>;
template class NetCDFSecondOrderStructureFunctionWriter<2, detail::SF2Approx>;
template class NetCDFSecondOrderStructureFunctionWriter<3, detail::SF2Approx>;

}
