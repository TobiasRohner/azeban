#ifndef AZEBAN_IO_WRITER_HPP_
#define AZEBAN_IO_WRITER_HPP_

#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <zisa/memory/array_view.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/communicator.hpp>
#endif

namespace azeban {

template <int Dim>
class Writer {
public:
  Writer(const Grid<Dim> &grid,
         const std::vector<real_t> &snapshot_times,
         zisa::int_t sample_idx_start = 0);
  Writer(const Writer &) = default;
  Writer &operator=(const Writer &) = default;

  virtual ~Writer() = default;

  virtual void reset();
  void set_snapshot_idx(zisa::int_t idx);
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
  Grid<Dim> grid_;
  std::vector<real_t> snapshot_times_;
  zisa::int_t sample_idx_;
  zisa::int_t snapshot_idx_;
};

}

#endif
