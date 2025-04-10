#include <azeban/io/netcdf_writer.hpp>

namespace azeban {

template <int Dim>
NetCDFWriter<Dim>::NetCDFWriter(int ncid,
                                const Grid<Dim> &grid,
                                const std::vector<double> &snapshot_times,
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
double NetCDFWriter<Dim>::next_timestep() const {
  if (snapshot_idx_ >= snapshot_times_.size()) {
    return std::numeric_limits<double>::infinity();
  }
  return snapshot_times_[snapshot_idx_];
}

template class NetCDFWriter<1>;
template class NetCDFWriter<2>;
template class NetCDFWriter<3>;

}
