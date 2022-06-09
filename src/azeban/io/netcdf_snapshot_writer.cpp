#include <azeban/io/netcdf_snapshot_writer.hpp>
#include <azeban/profiler.hpp>
#include <filesystem>
#include <zisa/io/netcdf_file.hpp>

namespace azeban {

template <int Dim>
NetCDFSnapshotWriter<Dim>::NetCDFSnapshotWriter(
    const std::string &path,
    bool store_u_hat,
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    zisa::int_t sample_idx_start,
    void *work_area)
    : super(grid, snapshot_times, sample_idx_start),
      path_(path),
      store_u_hat_(store_u_hat),
      work_area_(work_area) {
  std::filesystem::path sample_folder = path_;
  if (!std::filesystem::exists(sample_folder)) {
    std::filesystem::create_directories(sample_folder);
  }
}

template <int Dim>
void NetCDFSnapshotWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u, real_t t) {
  if (!store_u_hat_) {
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
      zisa::array_const_view<real_t, Dim> v_view(
          slice_shape,
          u.raw() + zisa::product(slice_shape),
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
}

template <int Dim>
void NetCDFSnapshotWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &u_hat, real_t t) {
  if (store_u_hat_) {
    ProfileHost pofile("NetCDFSnapshotWriter::write");
    LOG_ERR_IF(u_hat.memory_location() != zisa::device_type::cpu,
               "u_hat must be on the host");
    const auto compute_abs
        = [&](const zisa::array_const_view<complex_t, Dim> &hat,
              const zisa::array_view<real_t, Dim> &out) {
            for (zisa::int_t i = 0; i < hat.size(); ++i) {
              out[i] = abs(hat[i]);
            }
          };
    const auto compute_arg
        = [&](const zisa::array_const_view<complex_t, Dim> &hat,
              const zisa::array_view<real_t, Dim> &out) {
            for (zisa::int_t i = 0; i < hat.size(); ++i) {
              out[i] = arg(hat[i]);
            }
          };
    auto writer = make_writer(u_hat.shape(1), u_hat.shape(0));
    writer.write_scalar(t, "t");
    zisa::shape_t<Dim> slice_shape;
    for (int d = 0; d < Dim; ++d) {
      slice_shape[d] = u_hat.shape(d + 1);
    }
    zisa::array<real_t, Dim> out(slice_shape, zisa::device_type::cpu);
    zisa::array_const_view<complex_t, Dim> u_view(
        slice_shape, u_hat.raw(), zisa::device_type::cpu);
    compute_abs(u_view, out.view());
    zisa::save(writer, out, "u_hat_abs");
    compute_arg(u_view, out.view());
    zisa::save(writer, out, "u_hat_arg");
    if (Dim > 1) {
      zisa::array_const_view<complex_t, Dim> v_view(
          slice_shape,
          u_hat.raw() + zisa::product(slice_shape),
          zisa::device_type::cpu);
      compute_abs(v_view, out.view());
      zisa::save(writer, out, "v_hat_abs");
      compute_arg(v_view, out.view());
      zisa::save(writer, out, "v_hat_arg");
    }
    if (Dim > 2) {
      zisa::array_const_view<complex_t, Dim> w_view(
          slice_shape,
          u_hat.raw() + 2 * zisa::product(slice_shape),
          zisa::device_type::cpu);
      compute_abs(w_view, out.view());
      zisa::save(writer, out, "w_hat_abs");
      compute_arg(w_view, out.view());
      zisa::save(writer, out, "w_hat_arg");
    }
    if (u_hat.shape(0) == Dim + 1) {
      zisa::array_const_view<complex_t, Dim> rho_view(
          slice_shape,
          u_hat.raw() + Dim * zisa::product(slice_shape),
          zisa::device_type::cpu);
      compute_abs(rho_view, out.view());
      zisa::save(writer, out, "rho_hat_abs");
      compute_arg(rho_view, out.view());
      zisa::save(writer, out, "rho_hat_arg");
    }
    ++snapshot_idx_;
  }
}

#if AZEBAN_HAS_MPI
template <int Dim>
void NetCDFSnapshotWriter<Dim>::write(
    const zisa::array_const_view<real_t, Dim + 1> &u,
    real_t t,
    const Communicator *comm) {
  if (!store_u_hat_) {
    ProfileHost profile("SnapshotWriter::write");
    LOG_ERR_IF(u.memory_location() != zisa::device_type::cpu,
               "u must be on the host");
    const int rank = comm->rank();
    const int size = comm->size();

    std::vector<int> cnts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
      cnts[r]
          = zisa::pow<Dim - 1>(grid_.N_phys)
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
}

template <int Dim>
void NetCDFSnapshotWriter<Dim>::write(
    const zisa::array_const_view<complex_t, Dim + 1> &,
    real_t,
    const Communicator *) {
  if (store_u_hat_) {
    LOG_ERR("Not yet implemented");
  }
}
#endif

template <int Dim>
zisa::NetCDFSerialWriter
NetCDFSnapshotWriter<Dim>::make_writer(zisa::int_t N, zisa::int_t n_vars) {
  using nc_dim_t = std::tuple<std::string, size_t>;
  using nc_var_t
      = std::tuple<std::string, std::vector<std::string>, zisa::ErasedDataType>;
  std::vector<nc_dim_t> dims;
  std::vector<nc_var_t> vars;
  dims.emplace_back("t", 1);
  std::vector<std::string> tdim(1, "t");
  vars.emplace_back("t", tdim, zisa::erase_data_type<real_t>());
  dims.emplace_back("N", N);
  if (store_u_hat_) {
    dims.emplace_back("Nf", N / 2 + 1);
    std::vector<std::string> vardim(Dim, "N");
    vardim.back() = "Nf";
    vars.emplace_back("u_hat_abs", vardim, zisa::erase_data_type<real_t>());
    vars.emplace_back("u_hat_arg", vardim, zisa::erase_data_type<real_t>());
    if (Dim > 1) {
      vars.emplace_back("v_hat_abs", vardim, zisa::erase_data_type<real_t>());
      vars.emplace_back("v_hat_arg", vardim, zisa::erase_data_type<real_t>());
    }
    if (Dim > 2) {
      vars.emplace_back("w_hat_abs", vardim, zisa::erase_data_type<real_t>());
      vars.emplace_back("w_hat_arg", vardim, zisa::erase_data_type<real_t>());
    }
    if (n_vars == Dim + 1) {
      vars.emplace_back("rho_hat_abs", vardim, zisa::erase_data_type<real_t>());
      vars.emplace_back("rho_hat_arg", vardim, zisa::erase_data_type<real_t>());
    }
  } else {
    std::vector<std::string> vardim(Dim, "N");
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
