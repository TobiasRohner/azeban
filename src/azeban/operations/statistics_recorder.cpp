#include <azeban/operations/statistics_recorder.hpp>

namespace azeban {

template <int Dim>
StatisticsRecorder<Dim>::StatisticsRecorder(const Grid<Dim> &grid,
                                            bool has_tracer)
    : grid_(grid),
      has_tracer_(has_tracer),
      count_(0),
      Mk_(grid.shape_phys(Dim + has_tracer), zisa::device_type::cpu),
      Sk_(grid.shape_phys(Dim + has_tracer), zisa::device_type::cpu),
      is_finalized_(false) {
  zisa::fill(Mk_, real_t{0});
  zisa::fill(Sk_, real_t{0});
}

#if AZEBAN_HAS_MPI
template <int Dim>
StatisticsRecorder<Dim>::StatisticsRecorder(const Grid<Dim> &grid,
                                            bool has_tracer,
                                            Communicator *comm)
    : grid_(grid),
      has_tracer_(has_tracer),
      count_(0),
      Mk_(grid.shape_phys(Dim + has_tracer, comm), zisa::device_type::cpu),
      Sk_(grid.shape_phys(Dim + has_tracer, comm), zisa::device_type::cpu),
      is_finalized_(false) {
  zisa::fill(Mk_, real_t{0});
  zisa::fill(Sk_, real_t{0});
}
#endif

template <int Dim>
void StatisticsRecorder<Dim>::update(
    const zisa::array_const_view<real_t, Dim + 1> &u) {
  LOG_ERR_IF(is_finalized_,
             "Cannot update an already finalized statistics recorder");
  count_ += 1;
  for (zisa::int_t i = 0; i < u.size(); ++i) {
    const real_t u_new = u[i];
    real_t mean = Mk_[i];
    const real_t delta1 = u_new - mean;
    mean += delta1 / count_;
    const real_t delta2 = u_new - mean;
    Mk_[i] = mean;
    Sk_[i] += delta1 * delta2;
  }
}

template <int Dim>
void StatisticsRecorder<Dim>::finalize() {
  LOG_ERR_IF(is_finalized_,
             "Cannot finalize an already finalized statistics recorder");
  for (zisa::int_t i = 0; i < Sk_.size(); ++i) {
    Sk_[i] /= count_;
  }
  is_finalized_ = true;
}

template <int Dim>
zisa::array_const_view<real_t, Dim + 1> StatisticsRecorder<Dim>::mean() const {
  LOG_ERR_IF(
      !is_finalized_,
      "Please call StatisticsRecorder.finalize() before accessing the mean");
  return Mk_;
}

template <int Dim>
zisa::array_const_view<real_t, Dim + 1>
StatisticsRecorder<Dim>::variance() const {
  LOG_ERR_IF(!is_finalized_,
             "Please call StatisticsRecorder.finalize() before accessing the "
             "variance");
  return Sk_;
}

template class StatisticsRecorder<1>;
template class StatisticsRecorder<2>;
template class StatisticsRecorder<3>;

}
