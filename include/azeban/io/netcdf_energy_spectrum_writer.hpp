#ifndef AZEBAN_NETCDF_ENERGY_SPECTRUM_WRITER_HPP_
#define AZEBAN_NETCDF_ENERGY_SPECTRUM_WRITER_HPP_

#include <azeban/grid.hpp>
#include <azeban/io/netcdf_writer.hpp>
#include <chrono>

namespace azeban {

template <int Dim>
class NetCDFEnergySpectrumWriter : public NetCDFWriter<Dim> {
  using super = NetCDFWriter<Dim>;

public:
  NetCDFEnergySpectrumWriter(int ncid,
                             const Grid<Dim> &grid,
                             const std::vector<double> &snapshot_times,
                             int sample_idx_start);
  virtual ~NetCDFEnergySpectrumWriter() override = default;

protected:
  using super::grid_;
  using super::ncid_;
  using super::sample_idx_;
  using super::snapshot_idx_;

  virtual void write(const zisa::array_const_view<real_t, Dim + 1> &u,
                     double t) override;
  virtual void write(const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                     double t) override;
#if AZEBAN_HAS_MPI
  virtual void write(const zisa::array_const_view<real_t, Dim + 1> &u,
                     double t,
                     const Communicator *comm) override;
  virtual void write(const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                     double t,
                     const Communicator *comm) override;
#endif

private:
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
  int grpid_;
  int varid_real_time_;
  int varid_ek_;
};

}

#endif
