#include <azeban/io/absolute_longitudinal_structure_function_writer.hpp>

namespace azeban {

template <int Dim>
ANY_DEVICE AbsoluteLongitudinalStructureFunctionWriter<Dim>::
    AbsoluteLongitudinalStructureFunctionWriter(
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
            "S_par_abs",
            SFAbsoluteLongitudinalFunctional(p),
            max_h) {}

template class AbsoluteLongitudinalStructureFunctionWriter<1>;
template class AbsoluteLongitudinalStructureFunctionWriter<2>;
template class AbsoluteLongitudinalStructureFunctionWriter<3>;

}
