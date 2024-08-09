#ifndef AZEBAN_OPERATIONS_STATISTICS_RECORDER_HPP_
#define AZEBAN_OPERATIONS_STATISTICS_RECORDER_HPP_

#include <azeban/config.hpp>
#include <azeban/grid.hpp>
#include <zisa/memory/array.hpp>
#if AZEBAN_HAS_MPI
#include <azeban/mpi/communicator.hpp>
#endif

namespace azeban {

template <int Dim>
class StatisticsRecorder {
public:
  StatisticsRecorder(const Grid<Dim> &grid, bool has_tracer);
#if AZEBAN_HAS_MPI
  StatisticsRecorder(const Grid<Dim> &grid,
                     bool has_tracer,
                     Communicator *comm);
#endif

  void update(const zisa::array_const_view<real_t, Dim + 1> &u);
  void finalize();
#if AZEBAN_HAS_MPI
  void finalize(MPI_Comm comm);
#endif

  zisa::array_const_view<real_t, Dim + 1> mean() const;
  zisa::array_const_view<real_t, Dim + 1> variance() const;

private:
  Grid<Dim> grid_;
  bool has_tracer_;
  zisa::int_t count_;
  zisa::array<real_t, Dim + 1> Mk_;
  zisa::array<real_t, Dim + 1> Sk_;
  bool is_finalized_;
};

}

#endif
