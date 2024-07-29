#include <azeban/io/netcdf_writer.hpp>

namespace azeban {

template <int Dim>
NetCDFWriter<Dim>::NetCDFWriter(int ncid,
                                const Grid<Dim> &grid,
                                const std::vector<real_t> &snapshot_times,
                                zisa::int_t sample_idx_start)
    : ncid_(ncid),
      grid_(grid),
      snapshot_times_(snapshot_times),
      sample_idx_start_(sample_idx_start),
      snapshot_idx_(0),
      sample_idx_(sample_idx_start) {}

template <int Dim>
void NetCDFWriter<Dim>::reset() {
  snapshot_idx_ = 0;
  ++sample_idx_;
}

template <int Dim>
real_t NetCDFWriter<Dim>::next_timestep() const {
  if (snapshot_idx_ >= snapshot_times_.size()) {
    return std::numeric_limits<real_t>::infinity();
  }
  return snapshot_times_[snapshot_idx_];
}

template class NetCDFWriter<1>;
template class NetCDFWriter<2>;
template class NetCDFWriter<3>;

}
