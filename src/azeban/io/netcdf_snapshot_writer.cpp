#include <azeban/io/netcdf_snapshot_writer.hpp>
#include <azeban/profiler.hpp>
#include <filesystem>
#include <limits>
#include <zisa/io/netcdf_file.hpp>

namespace azeban {

template <int Dim>
NetCDFSnapshotWriter<Dim>::NetCDFSnapshotWriter(
    const std::string &path,
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    zisa::int_t sample_idx_start,
    void *work_area)
    : path_(path),
      grid_(grid),
      snapshot_times_(snapshot_times),
      sample_idx_(sample_idx_start),
      snapshot_idx_(0),
      work_area_(work_area) {
  std::filesystem::path sample_folder = path_;
  if (!std::filesystem::exists(sample_folder)) {
    std::filesystem::create_directories(sample_folder);
  }
}

template <int Dim>
void NetCDFSnapshotWriter<Dim>::reset() {
  snapshot_idx_ = 0;
  ++sample_idx_;
}

template <int Dim>
real_t NetCDFSnapshotWriter<Dim>::next_timestep() const {
  if (snapshot_idx_ >= snapshot_times_.size()) {
    return std::numeric_limits<real_t>::infinity();
  }
  return snapshot_times_[snapshot_idx_];
}

template <int Dim>
void NetCDFSnapshotWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u, real_t t) {
  ProfileHost pofile("NetCDFSnapshotWriter::write");
  LOG_ERR_IF(u.memory_location() != zisa::device_type::cpu,
             "u must be on the host");
  auto writer = make_writer(u.shape(1), u.shape(0));
  writer.write_scalar(t, "t");
  zisa::shape_t<Dim> slice_shape;
  for (int d = 0; d < Dim; ++d) {
    slice_shape[d] = u.shape(d + 1);
  }
  zisa::array_const_view<real_t, Dim> u_view(
      slice_shape, u.raw(), zisa::device_type::cpu);
  zisa::save(writer, u_view, "u");
  if (Dim > 1) {
    zisa::array_const_view<real_t, Dim> v_view(slice_shape,
                                               u.raw()
                                                   + zisa::product(slice_shape),
                                               zisa::device_type::cpu);
    zisa::save(writer, v_view, "v");
  }
  if (Dim > 2) {
    zisa::array_const_view<real_t, Dim> w_view(
        slice_shape,
        u.raw() + 2 * zisa::product(slice_shape),
        zisa::device_type::cpu);
    zisa::save(writer, w_view, "w");
  }
  if (u.shape(0) == Dim + 1) {
    zisa::array_const_view<real_t, Dim> rho_view(
        slice_shape,
        u.raw() + Dim * zisa::product(slice_shape),
        zisa::device_type::cpu);
    zisa::save(writer, rho_view, "rho");
  }
  ++snapshot_idx_;
}

template <int Dim>
void NetCDFSnapshotWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &, real_t) {}

#if AZEBAN_HAS_MPI
template <int Dim>
void NetCDFSnapshotWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u,
    real_t t,
    const Communicator *comm) {
  ProfileHost profile("SnapshotWriter::write");
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

  zisa::array<real_t, Dim + 1> u_init;
  if (rank == 0) {
    u_init = grid_.make_array_phys(u.shape(0), zisa::device_type::cpu);
  }
  std::vector<MPI_Request> reqs(u.shape(0));
  for (zisa::int_t i = 0; i < u.shape(0); ++i) {
    MPI_Igatherv(u.raw() + i * n_elems_per_component_loc,
                 cnts[rank],
                 mpi_type<real_t>(),
                 u_init.raw() + i * n_elems_per_component_glob,
                 cnts.data(),
                 displs.data(),
                 mpi_type<real_t>(),
                 0,
                 comm->get_mpi_comm(),
                 &reqs[i]);
  }
  MPI_Waitall(u.shape(0), reqs.data(), MPI_STATUSES_IGNORE);

  if (rank == 0) {
    auto writer = make_writer(u_init.shape(1), u_init.shape(0));
    writer.write_scalar(t, "t");
    zisa::shape_t<Dim> slice_shape;
    for (int d = 0; d < Dim; ++d) {
      slice_shape[d] = u_init.shape(d + 1);
    }
    zisa::array_const_view<real_t, Dim> u_view(
        slice_shape, u_init.raw(), zisa::device_type::cpu);
    zisa::save(writer, u_view, "u");
    if (Dim > 1) {
      zisa::array_const_view<real_t, Dim> v_view(
          slice_shape,
          u_init.raw() + zisa::product(slice_shape),
          zisa::device_type::cpu);
      zisa::save(writer, v_view, "v");
    }
    if (Dim > 2) {
      zisa::array_const_view<real_t, Dim> w_view(
          slice_shape,
          u_init.raw() + 2 * zisa::product(slice_shape),
          zisa::device_type::cpu);
      zisa::save(writer, w_view, "w");
    }
    if (u.shape(0) == Dim + 1) {
      zisa::array_const_view<real_t, Dim> rho_view(
          slice_shape,
          u_init.raw() + Dim * zisa::product(slice_shape),
          zisa::device_type::cpu);
      zisa::save(writer, rho_view, "rho");
    }
  }
  ++snapshot_idx_;
}

template <int Dim>
void NetCDFSnapshotWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &,
    real_t,
    const Communicator *) {}
#endif

template <int Dim>
zisa::NetCDFSerialWriter
NetCDFSnapshotWriter<Dim>::make_writer(zisa::int_t N, zisa::int_t n_vars) {
  using nc_dim_t = std::tuple<std::string, size_t>;
  using nc_var_t
      = std::tuple<std::string, std::vector<std::string>, zisa::ErasedDataType>;
  std::vector<nc_dim_t> dims;
  std::vector<nc_var_t> vars;
  dims.emplace_back("N", N);
  dims.emplace_back("t", 1);
  std::vector<std::string> vardim(Dim, "N");
  std::vector<std::string> tdim(1, "t");
  vars.emplace_back("t", tdim, zisa::erase_data_type<real_t>());
  vars.emplace_back("u", vardim, zisa::erase_data_type<real_t>());
  if (Dim > 1) {
    vars.emplace_back("v", vardim, zisa::erase_data_type<real_t>());
  }
  if (Dim > 2) {
    vars.emplace_back("w", vardim, zisa::erase_data_type<real_t>());
  }
  if (n_vars == Dim + 1) {
    vars.emplace_back("rho", vardim, zisa::erase_data_type<real_t>());
  }
  zisa::NetCDFFileStructure file_structure(dims, vars);

  const std::string file_name = path_ + "/" + "sample_"
                                + std::to_string(sample_idx_) + "_time_"
                                + std::to_string(snapshot_idx_) + ".nc";

  return zisa::NetCDFSerialWriter(file_name, file_structure);
}

template class NetCDFSnapshotWriter<1>;
template class NetCDFSnapshotWriter<2>;
template class NetCDFSnapshotWriter<3>;

}
