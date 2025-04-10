#include <azeban/io/netcdf_structure_function_cube_writer.hpp>
#include <azeban/netcdf.hpp>
#include <azeban/operations/structure_function.hpp>
#include <azeban/operations/structure_function_functionals.hpp>

namespace azeban {

template <int Dim, typename SF>
NetCDFStructureFunctionCubeWriter<Dim, SF>::NetCDFStructureFunctionCubeWriter(
    int ncid,
    const Grid<Dim> &grid,
    const std::vector<double> &snapshot_times,
    const std::string &name,
    const SF &func,
    zisa::int_t max_h,
    int sample_idx_start)
    : super(ncid, grid, snapshot_times, sample_idx_start),
      func_(func),
      max_h_(max_h) {
  // Define own group
  CHECK_NETCDF(nc_def_grp(ncid_, name.c_str(), &grpid_));
  // Define dimensions
  int dimid_member;
  int dimid_time;
  int dimid_r;
  CHECK_NETCDF(nc_inq_dimid(ncid_, "member", &dimid_member));
  CHECK_NETCDF(nc_def_dim(grpid_, "time", snapshot_times.size(), &dimid_time));
  CHECK_NETCDF(nc_def_dim(grpid_, "r", max_h_, &dimid_r));
  // Define variables
  int varid_sim_time;
  int varid_r;
  CHECK_NETCDF(
      nc_def_var(grpid_, "time", NC_DOUBLE, 1, &dimid_time, &varid_sim_time));
  CHECK_NETCDF(nc_def_var(grpid_, "r", NC_REAL, 1, &dimid_r, &varid_r));
  const int dimids_real_time[2] = {dimid_member, dimid_time};
  CHECK_NETCDF(nc_def_var(
      grpid_, "real_time", NC_DOUBLE, 2, dimids_real_time, &varid_real_time_));
  CHECK_NETCDF(
      nc_put_att_text(grpid_, varid_real_time_, "units", 8, "seconds"));
  const int dimids_S2[3] = {dimid_member, dimid_time, dimid_r};
  const size_t S2_chunksizes[3] = {1, 1, max_h_};
  CHECK_NETCDF(nc_def_var(grpid_, "SF", NC_REAL, 3, dimids_S2, &varid_S2_));
  CHECK_NETCDF(
      nc_def_var_chunking(grpid_, varid_S2_, NC_CHUNKED, S2_chunksizes));
  // Initialize variables
  CHECK_NETCDF(nc_put_var(grpid_, varid_sim_time, snapshot_times.data()));
  std::vector<real_t> r(max_h_);
  for (zisa::int_t i = 0; i < max_h_; ++i) {
    r[i] = static_cast<real_t>(i) / grid.N_phys;
  }
  CHECK_NETCDF(nc_put_var(grpid_, varid_r, r.data()));
  // Record the current time
  start_time_ = std::chrono::steady_clock::now();
}

template <int Dim, typename SF>
void NetCDFStructureFunctionCubeWriter<Dim, SF>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u, double) {
  const std::vector<real_t> sf = structure_function<Dim>(u, max_h_, func_);
  const size_t start[3] = {sample_idx_, snapshot_idx_, 0};
  const size_t count[3] = {1, 1, max_h_};
  CHECK_NETCDF(nc_put_vara(grpid_, varid_S2_, start, count, sf.data()));
  const auto time = std::chrono::steady_clock::now();
  const auto elapsed
      = std::chrono::duration_cast<std::chrono::duration<double>>(
          time - start_time_);
  const size_t index[2] = {sample_idx_, snapshot_idx_};
  const double elapsed_count = elapsed.count();
  CHECK_NETCDF(nc_put_var1(grpid_, varid_real_time_, index, &elapsed_count));
  ++snapshot_idx_;
}

template <int Dim, typename SF>
void NetCDFStructureFunctionCubeWriter<Dim, SF>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &, double) {
  // Nothing to do here
}

#if AZEBAN_HAS_MPI
template <int Dim, typename SF>
void NetCDFStructureFunctionCubeWriter<Dim, SF>::write(
    const zisa::array_const_view<real_t, Dim + 1> &,
    double,
    const Communicator *) {
  // Nothing to do here
}

template <int Dim, typename SF>
void NetCDFStructureFunctionCubeWriter<Dim, SF>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &,
    double,
    const Communicator *) {
  AZEBAN_ERR("Structure Function Cube does not support MPI");
}
#endif

template class NetCDFStructureFunctionCubeWriter<1, SFCubeFunctional>;
template class NetCDFStructureFunctionCubeWriter<2, SFCubeFunctional>;
template class NetCDFStructureFunctionCubeWriter<3, SFCubeFunctional>;
template class NetCDFStructureFunctionCubeWriter<1, SFThirdOrderFunctional>;
template class NetCDFStructureFunctionCubeWriter<2, SFThirdOrderFunctional>;
template class NetCDFStructureFunctionCubeWriter<3, SFThirdOrderFunctional>;
template class NetCDFStructureFunctionCubeWriter<1, SFLongitudinalFunctional>;
template class NetCDFStructureFunctionCubeWriter<2, SFLongitudinalFunctional>;
template class NetCDFStructureFunctionCubeWriter<3, SFLongitudinalFunctional>;
template class NetCDFStructureFunctionCubeWriter<
    1,
    SFAbsoluteLongitudinalFunctional>;
template class NetCDFStructureFunctionCubeWriter<
    2,
    SFAbsoluteLongitudinalFunctional>;
template class NetCDFStructureFunctionCubeWriter<
    3,
    SFAbsoluteLongitudinalFunctional>;

}
