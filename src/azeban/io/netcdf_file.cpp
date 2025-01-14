#include <azeban/io/netcdf_file.hpp>
#include <azeban/netcdf.hpp>
#include <azeban/profiler.hpp>
#include <experimental/filesystem>
#if AZEBAN_HAS_MPI
#include <mpi.h>
#endif

namespace azeban {

template <int Dim>
NetCDFFile<Dim>::NetCDFFile(const std::string &path,
                            const Grid<Dim> &grid,
                            zisa::int_t num_samples,
                            zisa::int_t sample_idx_start,
                            const std::string &config,
                            const std::string &script)
    : super(grid, std::vector<real_t>(), sample_idx_start) {
  // Open new NetCDF4 file
#if AZEBAN_HAS_MPI
  int size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
  // Define universally useful dimensions
  CHECK_NETCDF(nc_def_dim(ncid_, "member", num_samples, &dimid_member_));
  CHECK_NETCDF(nc_def_dim(ncid_, "x", grid.N_phys, &dimid_dim_[0]));
  if constexpr (Dim > 1) {
    CHECK_NETCDF(nc_def_dim(ncid_, "y", grid.N_phys, &dimid_dim_[1]));
  }
  if constexpr (Dim > 2) {
    CHECK_NETCDF(nc_def_dim(ncid_, "z", grid.N_phys, &dimid_dim_[2]));
  }
  // Add attributes describing the simulation setup
  CHECK_NETCDF(nc_put_att_text(
      ncid_, NC_GLOBAL, "config", config.size() + 1, config.c_str()));
  if (script.size() > 0) {
    CHECK_NETCDF(nc_put_att_text(
        ncid_, NC_GLOBAL, "init_script", script.size() + 1, script.c_str()));
  }
  // Define universal variables
  int varid_member;
  int varid_x;
  int varid_y;
  int varid_z;
  CHECK_NETCDF(
      nc_def_var(ncid_, "member", NC_INT, 1, &dimid_member_, &varid_member));
  CHECK_NETCDF(nc_def_var(ncid_, "x", NC_REAL, 1, &dimid_dim_[0], &varid_x));
  if constexpr (Dim > 1) {
    CHECK_NETCDF(nc_def_var(ncid_, "y", NC_REAL, 1, &dimid_dim_[1], &varid_y));
  }
  if constexpr (Dim > 2) {
    CHECK_NETCDF(nc_def_var(ncid_, "z", NC_REAL, 1, &dimid_dim_[2], &varid_z));
  }
  // Initialize universal variables
  std::vector<int> member(num_samples);
  for (zisa::int_t i = 0; i < num_samples; ++i) {
    member[i] = i;
  }
  std::vector<real_t> x(grid.N_phys);
  for (zisa::int_t i = 0; i < grid.N_phys; ++i) {
    x[i] = static_cast<real_t>(i) / grid.N_phys;
  }
  CHECK_NETCDF(nc_put_var(ncid_, varid_member, member.data()));
  CHECK_NETCDF(nc_put_var(ncid_, varid_x, x.data()));
  if constexpr (Dim > 1) {
    CHECK_NETCDF(nc_put_var(ncid_, varid_y, x.data()));
  }
  if constexpr (Dim > 2) {
    CHECK_NETCDF(nc_put_var(ncid_, varid_z, x.data()));
  }
}

template <int Dim>
NetCDFFile<Dim>::~NetCDFFile() {
  writers_.clear();
#if AZEBAN_HAS_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  nc_close(ncid_);
}

template <int Dim>
int NetCDFFile<Dim>::ncid() const {
  return ncid_;
}

template <int Dim>
void NetCDFFile<Dim>::add_writer(std::unique_ptr<NetCDFWriter<Dim>> &&writer) {
  writers_.push_back(std::move(writer));
}

template <int Dim>
void NetCDFFile<Dim>::reset() {
  for (auto &writer : writers_) {
    writer->reset();
  }
}

template <int Dim>
real_t NetCDFFile<Dim>::next_timestep() const {
  real_t t = writers_[0]->next_timestep();
  for (size_t i = 1; i < writers_.size(); ++i) {
    t = zisa::min(t, writers_[i]->next_timestep());
  }
  return t;
}

template <int Dim>
void NetCDFFile<Dim>::write(const zisa::array_const_view<real_t, Dim + 1> &u,
                            real_t t) {
  for (auto &writer : writers_) {
    if (writer->next_timestep() == t) {
      writer->write(u, t);
    }
  }
}

template <int Dim>
void NetCDFFile<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat, real_t t) {
  for (auto &writer : writers_) {
    if (writer->next_timestep() == t) {
      writer->write(u_hat, t);
    }
  }
}

#if AZEBAN_HAS_MPI
template <int Dim>
void NetCDFFile<Dim>::write(const zisa::array_const_view<real_t, Dim + 1> &u,
                            real_t t,
                            const Communicator *comm) {
  for (auto &writer : writers_) {
    if (writer->next_timestep() == t) {
      writer->write(u, t, comm);
    }
  }
}

template <int Dim>
void NetCDFFile<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
    real_t t,
    const Communicator *comm) {
  for (auto &writer : writers_) {
    if (writer->next_timestep() == t) {
      writer->write(u_hat, t, comm);
    }
  }
}
#endif

template class NetCDFFile<1>;
template class NetCDFFile<2>;
template class NetCDFFile<3>;

}
