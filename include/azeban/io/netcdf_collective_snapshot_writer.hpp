#ifndef AZEBAN_IO_NETCDF_COLLECTIVE_SNAPSHOT_WRITER_HPP_
#define AZEBAN_IO_NETCDF_COLLECTIVE_SNAPSHOT_WRITER_HPP_

#include <azeban/grid.hpp>
#include <azeban/io/writer.hpp>
#include <map>
#include <vector>

namespace azeban {

template <int Dim>
class NetCDFCollectiveSnapshotWriter : public Writer<Dim> {
  using super = Writer<Dim>;

public:
  NetCDFCollectiveSnapshotWriter(const std::string &path,
                                 const Grid<Dim> &grid,
                                 const std::vector<real_t> &snapshot_times,
                                 bool has_tracer,
                                 zisa::int_t num_samples,
                                 zisa::int_t sample_idx_start,
                                 void *work_area = nullptr);
  NetCDFCollectiveSnapshotWriter(const NetCDFCollectiveSnapshotWriter &)
      = default;
  NetCDFCollectiveSnapshotWriter(NetCDFCollectiveSnapshotWriter &&) = default;

  virtual ~NetCDFCollectiveSnapshotWriter() override;

  NetCDFCollectiveSnapshotWriter &
  operator=(const NetCDFCollectiveSnapshotWriter &)
      = default;
  NetCDFCollectiveSnapshotWriter &operator=(NetCDFCollectiveSnapshotWriter &&)
      = default;

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
  int ncid_;
  int dimids_[Dim + 2];
  int varids_[Dim + 1];
  void *work_area_;

  void setup_file(const Grid<Dim> &grid,
                  const std::vector<real_t> &snapshot_times,
                  bool has_tracer,
                  zisa::int_t num_samples);
};

}

#endif
