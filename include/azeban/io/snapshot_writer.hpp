#ifndef AZEBAN_IO_SNAPSHOT_WRITER_HPP_
#define AZEBAN_IO_SNAPSHOT_WRITER_HPP_

#include <azeban/grid.hpp>
#include <azeban/simulation.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/communicator.hpp>
#endif

namespace azeban {

template <int Dim>
class SnapshotWriter {
public:
  SnapshotWriter() = default;
  SnapshotWriter(const SnapshotWriter &) = default;
  SnapshotWriter(SnapshotWriter &&) = default;
  virtual ~SnapshotWriter() = default;
  SnapshotWriter &operator=(const SnapshotWriter &) = default;
  SnapshotWriter &operator=(SnapshotWriter &&) = default;

  void write_snapshot(const Simulation<Dim> &sim, zisa::int_t sample_idx);
#if AZEBAN_HAS_MPI
  void write_snapshot(const Simulation<Dim> &sim,
                      zisa::int_t sample_idx,
                      const Communicator *comm);
#endif

protected:
  virtual void
  do_write_snapshot(zisa::int_t sample_idx,
                    real_t t,
                    const zisa::array_const_view<real_t, Dim + 1> &u)
      = 0;
};

}

#endif
