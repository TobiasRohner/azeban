#include <azeban/io/longitudinal_structure_function_writer.hpp>

namespace azeban {

template <int Dim>
ANY_DEVICE
LongitudinalStructureFunctionWriter<Dim>::LongitudinalStructureFunctionWriter(
    const std::string &path,
    const Grid<Dim> &grid,
    const std::vector<double> &snapshot_times,
    zisa::int_t sample_idx_start,
    real_t p,
    ssize_t max_h)
    : super(path,
            grid,
            snapshot_times,
            sample_idx_start,
            "S_par",
            SFLongitudinalFunctional(p),
            max_h) {}

template class LongitudinalStructureFunctionWriter<1>;
template class LongitudinalStructureFunctionWriter<2>;
template class LongitudinalStructureFunctionWriter<3>;

}
