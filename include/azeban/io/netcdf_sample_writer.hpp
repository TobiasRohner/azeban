#ifndef AZEBAN_NETCDF_SAMPLE_WRITER_HPP_
#define AZEBAN_NETCDF_SAMPLE_WRITER_HPP_

#include <azeban/grid.hpp>
#include <azeban/io/netcdf_writer.hpp>
#include <azeban/operations/fft_factory.hpp>
#include <azeban/operations/statistics_recorder.hpp>
#include <chrono>

namespace azeban {

template <int Dim>
class NetCDFSampleWriter : public NetCDFWriter<Dim> {
  using super = NetCDFWriter<Dim>;

public:
  NetCDFSampleWriter(int ncid,
                     const Grid<Dim> &grid,
                     zisa::int_t N,
                     const std::vector<real_t> &snapshot_times,
                     bool has_trcer,
                     bool store_mean_var,
                     zisa::int_t sample_idx_start);
  virtual ~NetCDFSampleWriter() override;

protected:
  using super::grid_;
  using super::ncid_;
  using super::sample_idx_;
  using super::sample_idx_start_;
  using super::snapshot_idx_;

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
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
  zisa::int_t N_;
  bool has_tracer_;
  bool store_mean_var_;
  int grpid_;
  int varid_real_time_;
  int varids_uvw_[Dim + 1];
  int grpid_mean_;
  int varids_mean_uvw_[Dim + 1];
  int grpid_var_;
  int varids_var_uvw_[Dim + 1];
  zisa::array<complex_t, Dim + 1> u_hat_down_;
  zisa::array<real_t, Dim + 1> u_down_;
  std::shared_ptr<FFT<Dim, real_t>> fft_down_;
  std::vector<StatisticsRecorder<Dim>> statistics_;

  void store_u(const zisa::array_const_view<real_t, Dim + 1> &u);
};

}

#endif
