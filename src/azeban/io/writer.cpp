#include <azeban/io/writer.hpp>
#include <limits>

namespace azeban {

template <int Dim>
Writer<Dim>::Writer(const Grid<Dim> &grid,
                    const std::vector<double> &snapshot_times,
                    zisa::int_t sample_idx_start)
    : grid_(grid),
      snapshot_times_(snapshot_times),
      sample_idx_(sample_idx_start),
      snapshot_idx_(0) {}

template <int Dim>
void Writer<Dim>::reset() {
  snapshot_idx_ = 0;
  ++sample_idx_;
}

template <int Dim>
void Writer<Dim>::set_snapshot_idx(zisa::int_t idx) {
  snapshot_idx_ = idx;
}

template <int Dim>
double Writer<Dim>::next_timestep() const {
  if (snapshot_idx_ >= snapshot_times_.size()) {
    return std::numeric_limits<double>::infinity();
  }
  return snapshot_times_[snapshot_idx_];
}

template class Writer<1>;
template class Writer<2>;
template class Writer<3>;

}
