#ifndef AZEBAN_NETCDF_SEOND_ORDER_STRUCTURE_FUNCTION_WRITER_HPP_
#define AZEBAN_NETCDF_SEOND_ORDER_STRUCTURE_FUNCTION_WRITER_HPP_

#include <azeban/grid.hpp>
#include <azeban/io/netcdf_writer.hpp>
#include <azeban/io/second_order_structure_function_writer.hpp>
#include <chrono>

namespace azeban {

template <int Dim, typename SF>
class NetCDFSecondOrderStructureFunctionWriter : public NetCDFWriter<Dim> {
  using super = NetCDFWriter<Dim>;

public:
  NetCDFSecondOrderStructureFunctionWriter(int ncid,
                             const Grid<Dim> &grid,
                             const std::vector<real_t> &snapshot_times,
                             int sample_idx_start);
  virtual ~NetCDFSecondOrderStructureFunctionWriter() override = default;

protected:
  using super::grid_;
  using super::ncid_;
  using super::sample_idx_;
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
  int grpid_;
  int varid_real_time_;
  int varid_S2_;
};

}

#endif
