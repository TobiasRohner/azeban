#ifndef AZEBAN_IO_NETCDF_SNAPSHOT_WRITER_HPP_
#define AZEBAN_IO_NETCDF_SNAPSHOT_WRITER_HPP_

#include <azeban/io/snapshot_writer.hpp>
#include <map>
#include <zisa/io/netcdf_serial_writer.hpp>

namespace azeban {

template <int Dim>
class NetCDFSnapshotWriter : public SnapshotWriter<Dim> {
  using super = SnapshotWriter<Dim>;

public:
  NetCDFSnapshotWriter(const std::string &path);
  NetCDFSnapshotWriter(const NetCDFSnapshotWriter &) = default;
  NetCDFSnapshotWriter(NetCDFSnapshotWriter &&) = default;

  virtual ~NetCDFSnapshotWriter() override = default;

  NetCDFSnapshotWriter &operator=(const NetCDFSnapshotWriter &) = default;
  NetCDFSnapshotWriter &operator=(NetCDFSnapshotWriter &&) = default;

protected:
  virtual void
  do_write_snapshot(zisa::int_t sample_idx,
                    real_t t,
                    const zisa::array_const_view<real_t, Dim + 1> &u) override;

private:
  std::string path_;
  std::map<zisa::int_t, unsigned> snapshot_indices_;

  zisa::NetCDFSerialWriter
  make_writer(zisa::int_t sample_idx, zisa::int_t N, zisa::int_t n_vars);
};

}

#endif
