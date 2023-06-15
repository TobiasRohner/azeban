#include <azeban/io/structure_function_cube_writer.hpp>

namespace azeban {

template <int Dim>
ANY_DEVICE StructureFunctionCubeWriter<Dim>::StructureFunctionCubeWriter(
    const std::string &path,
    const Grid<Dim> &grid,
    const std::vector<real_t> &snapshot_times,
    zisa::int_t sample_idx_start,
    real_t p,
    ssize_t max_h)
    : super(path,
            grid,
            snapshot_times,
            sample_idx_start,
            "SF_Cube_" + std::to_string(p),
            SFCubeFunctional(p),
            max_h) {}

template class StructureFunctionCubeWriter<1>;
template class StructureFunctionCubeWriter<2>;
template class StructureFunctionCubeWriter<3>;

}
