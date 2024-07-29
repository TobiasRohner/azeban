#ifndef AZEBAN_IO_NETCDF_WRITER_HPP_
#define AZEBAN_IO_NETCDF_WRITER_HPP_

#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <azeban/mpi/communicator.hpp>
#include <vector>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <int Dim>
class NetCDFWriter {
public:
  NetCDFWriter(int ncid,
               const Grid<Dim> &grid,
               const std::vector<real_t> &snapshot_times,
               zisa::int_t sample_idx_start);
  NetCDFWriter(const NetCDFWriter &) = delete;
  NetCDFWriter(NetCDFWriter &&) = default;

  virtual ~NetCDFWriter() = default;

  NetCDFWriter &operator=(const NetCDFWriter &) = delete;
  NetCDFWriter &operator=(NetCDFWriter &&) = default;

  virtual void reset();
  virtual real_t next_timestep() const;
  virtual void write(const zisa::array_const_view<real_t, Dim + 1> &u, real_t t)
      = 0;
  virtual void write(const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                     real_t t)
      = 0;
#if AZEBAN_HAS_MPI
  virtual void write(const zisa::array_const_view<real_t, Dim + 1> &u,
                     real_t t,
                     const Communicator *comm)
      = 0;
  virtual void write(const zisa::array_const_view<complex_t, Dim + 1> &u_hat,
                     real_t t,
                     const Communicator *comm)
      = 0;
#endif

protected:
  int ncid_;
  Grid<Dim> grid_;
  std::vector<real_t> snapshot_times_;
  zisa::int_t sample_idx_start_;
  zisa::int_t snapshot_idx_;
  zisa::int_t sample_idx_;
};

}

#endif
