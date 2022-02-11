#include <azeban/io/netcdf_snapshot_writer.hpp>
#include <filesystem>
#include <zisa/io/netcdf_file.hpp>

namespace azeban {

template <int Dim>
NetCDFSnapshotWriter<Dim>::NetCDFSnapshotWriter(const std::string &path)
    : path_(path) {
  std::filesystem::path sample_folder = path_;
  if (!std::filesystem::exists(sample_folder)) {
    std::filesystem::create_directories(sample_folder);
  }
}

template <>
void NetCDFSnapshotWriter<1>::do_write_snapshot(
    zisa::int_t sample_idx,
    real_t t,
    const zisa::array_const_view<real_t, 2> &u) {
  auto writer = make_writer(sample_idx, u.shape(1), u.shape(0));
  writer.write_scalar(t, "t");
  zisa::shape_t<1> slice_shape(u.shape(1));
  zisa::array_const_view<real_t, 1> u_view(
      slice_shape, u.raw(), zisa::device_type::cpu);
  zisa::save(writer, u_view, "u");
  if (u.shape(0) == 2) {
    zisa::array_const_view<real_t, 1> rho_view(slice_shape,
                                               u.raw()
                                                   + zisa::product(slice_shape),
                                               zisa::device_type::cpu);
    zisa::save(writer, rho_view, "rho");
  }
}

template <>
void NetCDFSnapshotWriter<2>::do_write_snapshot(
    zisa::int_t sample_idx,
    real_t t,
    const zisa::array_const_view<real_t, 3> &u) {
  auto writer = make_writer(sample_idx, u.shape(1), u.shape(0));
  writer.write_scalar(t, "t");
  zisa::shape_t<2> slice_shape(u.shape(1), u.shape(2));
  zisa::array_const_view<real_t, 2> u_view(
      slice_shape, u.raw(), zisa::device_type::cpu);
  zisa::save(writer, u_view, "u");
  zisa::array_const_view<real_t, 2> v_view(slice_shape,
                                           u.raw() + zisa::product(slice_shape),
                                           zisa::device_type::cpu);
  zisa::save(writer, v_view, "v");
  if (u.shape(0) == 3) {
    zisa::array_const_view<real_t, 2> rho_view(
        slice_shape,
        u.raw() + 2 * zisa::product(slice_shape),
        zisa::device_type::cpu);
    zisa::save(writer, rho_view, "rho");
  }
}

template <>
void NetCDFSnapshotWriter<3>::do_write_snapshot(
    zisa::int_t sample_idx,
    real_t t,
    const zisa::array_const_view<real_t, 4> &u) {
  auto writer = make_writer(sample_idx, u.shape(1), u.shape(0));
  writer.write_scalar(t, "t");
  zisa::shape_t<3> slice_shape(u.shape(1), u.shape(2), u.shape(3));
  zisa::array_const_view<real_t, 3> u_view(
      slice_shape, u.raw(), zisa::device_type::cpu);
  zisa::save(writer, u_view, "u");
  zisa::array_const_view<real_t, 3> v_view(slice_shape,
                                           u.raw() + zisa::product(slice_shape),
                                           zisa::device_type::cpu);
  zisa::save(writer, v_view, "v");
  zisa::array_const_view<real_t, 3> w_view(slice_shape,
                                           u.raw()
                                               + 2 * zisa::product(slice_shape),
                                           zisa::device_type::cpu);
  zisa::save(writer, w_view, "w");
  if (u.shape(0) == 4) {
    zisa::array_const_view<real_t, 3> rho_view(
        slice_shape,
        u.raw() + 3 * zisa::product(slice_shape),
        zisa::device_type::cpu);
    zisa::save(writer, rho_view, "rho");
  }
}

template <int Dim>
zisa::NetCDFSerialWriter NetCDFSnapshotWriter<Dim>::make_writer(
    zisa::int_t sample_idx, zisa::int_t N, zisa::int_t n_vars) {
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

  const zisa::int_t snapshot_index = snapshot_indices_[sample_idx]++;
  const std::string file_name = path_ + "/" + "sample_"
                                + std::to_string(sample_idx) + "_time_"
                                + std::to_string(snapshot_index) + ".nc";

  return zisa::NetCDFSerialWriter(file_name, file_structure);
}

template class NetCDFSnapshotWriter<1>;
template class NetCDFSnapshotWriter<2>;
template class NetCDFSnapshotWriter<3>;

}
