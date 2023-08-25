#ifndef AZEBAN_CUDA_OPERATIONS_STRUCTURE_FUNCTION_CUDA_HPP_
#define AZEBAN_CUDA_OPERATIONS_STRUCTURE_FUNCTION_CUDA_HPP_

#include <azeban/config.hpp>
#include <azeban/operations/structure_function_functionals.hpp>
#include <vector>
#include <zisa/memory/array_view.hpp>

namespace azeban {

template <typename Function>
std::vector<real_t>
structure_function_cuda(const zisa::array_const_view<real_t, 2> &u,
                        ssize_t max_h,
                        const Function &func);
template <typename Function>
std::vector<real_t>
structure_function_cuda(const zisa::array_const_view<real_t, 3> &u,
                        ssize_t max_h,
                        const Function &func);
template <typename Function>
std::vector<real_t>
structure_function_cuda(const zisa::array_const_view<real_t, 4> &u,
                        ssize_t max_h,
                        const Function &func);

extern template std::vector<real_t> structure_function_cuda<SFCubeFunctional>(
    const zisa::array_const_view<real_t, 2> &,
    ssize_t,
    const SFCubeFunctional &);
extern template std::vector<real_t> structure_function_cuda<SFCubeFunctional>(
    const zisa::array_const_view<real_t, 3> &,
    ssize_t,
    const SFCubeFunctional &);
extern template std::vector<real_t> structure_function_cuda<SFCubeFunctional>(
    const zisa::array_const_view<real_t, 4> &,
    ssize_t,
    const SFCubeFunctional &);
extern template std::vector<real_t>
structure_function_cuda<SFThirdOrderFunctional>(
    const zisa::array_const_view<real_t, 2> &,
    ssize_t,
    const SFThirdOrderFunctional &);
extern template std::vector<real_t>
structure_function_cuda<SFThirdOrderFunctional>(
    const zisa::array_const_view<real_t, 3> &,
    ssize_t,
    const SFThirdOrderFunctional &);
extern template std::vector<real_t>
structure_function_cuda<SFThirdOrderFunctional>(
    const zisa::array_const_view<real_t, 4> &,
    ssize_t,
    const SFThirdOrderFunctional &);
extern template std::vector<real_t>
structure_function_cuda<SFLongitudinalFunctional>(
    const zisa::array_const_view<real_t, 2> &,
    ssize_t,
    const SFLongitudinalFunctional &);
extern template std::vector<real_t>
structure_function_cuda<SFLongitudinalFunctional>(
    const zisa::array_const_view<real_t, 3> &,
    ssize_t,
    const SFLongitudinalFunctional &);
extern template std::vector<real_t>
structure_function_cuda<SFLongitudinalFunctional>(
    const zisa::array_const_view<real_t, 4> &,
    ssize_t,
    const SFLongitudinalFunctional &);
extern template std::vector<real_t>
structure_function_cuda<SFAbsoluteLongitudinalFunctional>(
    const zisa::array_const_view<real_t, 2> &,
    ssize_t,
    const SFAbsoluteLongitudinalFunctional &);
extern template std::vector<real_t>
structure_function_cuda<SFAbsoluteLongitudinalFunctional>(
    const zisa::array_const_view<real_t, 3> &,
    ssize_t,
    const SFAbsoluteLongitudinalFunctional &);
extern template std::vector<real_t>
structure_function_cuda<SFAbsoluteLongitudinalFunctional>(
    const zisa::array_const_view<real_t, 4> &,
    ssize_t,
    const SFAbsoluteLongitudinalFunctional &);

}

#endif
