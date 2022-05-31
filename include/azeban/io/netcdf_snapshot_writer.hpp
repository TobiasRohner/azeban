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
                       const Grid<Dim> &grid,
                       const std::vector<real_t> &snapshot_times,
                       zisa::int_t sample_idx_start,
                       void *work_area = nullptr);
  NetCDFSnapshotWriter(const NetCDFSnapshotWriter &) = default;
  NetCDFSnapshotWriter(NetCDFSnapshotWriter &&) = default;

  virtual ~NetCDFSnapshotWriter() override = default;

  NetCDFSnapshotWriter &operator=(const NetCDFSnapshotWriter &) = default;
  NetCDFSnapshotWriter &operator=(NetCDFSnapshotWriter &&) = default;

  virtual void reset() override;
  virtual real_t next_timestep() const override;
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

private:
  std::string path_;
  Grid<Dim> grid_;
  std::vector<real_t> snapshot_times_;
  zisa::int_t sample_idx_;
  zisa::int_t snapshot_idx_;
  void *work_area_;

  zisa::NetCDFSerialWriter make_writer(zisa::int_t N, zisa::int_t n_vars);
};

}

#endif
