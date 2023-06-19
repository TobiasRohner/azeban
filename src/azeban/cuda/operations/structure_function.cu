#include <azeban/cuda/operations/structure_function_cuda_impl.cuh>

namespace azeban {

template std::vector<real_t> structure_function_cuda<SFCubeFunctional>(
    const zisa::array_const_view<real_t, 2> &,
    ssize_t,
    const SFCubeFunctional &);
template std::vector<real_t> structure_function_cuda<SFCubeFunctional>(
    const zisa::array_const_view<real_t, 3> &,
    ssize_t,
    const SFCubeFunctional &);
template std::vector<real_t> structure_function_cuda<SFCubeFunctional>(
    const zisa::array_const_view<real_t, 4> &,
    ssize_t,
    const SFCubeFunctional &);
template std::vector<real_t> structure_function_cuda<SFThirdOrderFunctional>(
    const zisa::array_const_view<real_t, 2> &,
    ssize_t,
    const SFThirdOrderFunctional &);
template std::vector<real_t> structure_function_cuda<SFThirdOrderFunctional>(
    const zisa::array_const_view<real_t, 3> &,
    ssize_t,
    const SFThirdOrderFunctional &);
template std::vector<real_t> structure_function_cuda<SFThirdOrderFunctional>(
    const zisa::array_const_view<real_t, 4> &,
    ssize_t,
    const SFThirdOrderFunctional &);
template std::vector<real_t> structure_function_cuda<SFLongitudinalFunctional>(
    const zisa::array_const_view<real_t, 2> &,
    ssize_t,
    const SFLongitudinalFunctional &);
template std::vector<real_t> structure_function_cuda<SFLongitudinalFunctional>(
    const zisa::array_const_view<real_t, 3> &,
    ssize_t,
    const SFLongitudinalFunctional &);
template std::vector<real_t> structure_function_cuda<SFLongitudinalFunctional>(
    const zisa::array_const_view<real_t, 4> &,
    ssize_t,
    const SFLongitudinalFunctional &);

}
