#include <azeban/io/third_order_structure_function_writer.hpp>

namespace azeban {

template <int Dim>
ANY_DEVICE
ThirdOrderStructureFunctionWriter<Dim>::ThirdOrderStructureFunctionWriter(
    const std::string &path,
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    zisa::int_t sample_idx_start,
    ssize_t max_h)
    : super(path,
            grid,
            snapshot_times,
            sample_idx_start,
            "S3",
            SFThirdOrderFunctional(),
            max_h) {}

template class ThirdOrderStructureFunctionWriter<1>;
template class ThirdOrderStructureFunctionWriter<2>;
template class ThirdOrderStructureFunctionWriter<3>;

}
