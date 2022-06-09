#ifndef AZEBAN_IO_NETCDF_SNAPSHOT_WRITER_HPP_
#define AZEBAN_IO_NETCDF_SNAPSHOT_WRITER_HPP_

#include <azeban/grid.hpp>
#include <azeban/io/writer.hpp>
#include <map>
#include <vector>
#include <zisa/io/netcdf_serial_writer.hpp>

namespace azeban {

template <int Dim>
class NetCDFSnapshotWriter : public Writer<Dim> {
  using super = Writer<Dim>;

public:
  NetCDFSnapshotWriter(const std::string &path,
                       bool store_u_hat,
                       const Grid<Dim> &grid,
                       const std::vector<real_t> &snapshot_times,
                       zisa::int_t sample_idx_start,
                       void *work_area = nullptr);
  NetCDFSnapshotWriter(const NetCDFSnapshotWriter &) = default;
  NetCDFSnapshotWriter(NetCDFSnapshotWriter &&) = default;

  virtual ~NetCDFSnapshotWriter() override = default;

  NetCDFSnapshotWriter &operator=(const NetCDFSnapshotWriter &) = default;
  NetCDFSnapshotWriter &operator=(NetCDFSnapshotWriter &&) = default;

  using super::next_timestep;
  using super::reset;
  virtual void write(const zisa::array_const_view<real_t, Dim + 1> &u,
                     real_t t) override;
  virtual void write(const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                     real_t t) override;
#if AZEBAN_HAS_MPI
  virtual void write(const zisa::array_const_view<real_t, Dim + 1> &u,
                     real_t t,
                     const Communicator *comm) override;
  virtual void write(const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                     real_t t,
                     const Communicator *comm) override;
#endif

protected:
  using super::grid_;
  using super::sample_idx_;
  using super::snapshot_idx_;
  using super::snapshot_times_;

private:
  std::string path_;
  bool store_u_hat_;
  void *work_area_;

  zisa::NetCDFSerialWriter make_writer(zisa::int_t N, zisa::int_t n_vars);
};

}

#endif
