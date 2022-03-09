#ifndef AZEBAN_CUDA_OPERATIONS_TRANSPOSE_HPP_
#define AZEBAN_CUDA_OPERATIONS_TRANSPOSE_HPP_

#include <azeban/config.hpp>
#include <zisa/memory/array_view.hpp>

namespace azeban {

void transpose_cuda_preprocess(const zisa::array_const_view<complex_t, 3> &from,
                               const zisa::array_view<complex_t, 4> &sendbuf,
                               const zisa::shape_t<3> *from_shapes,
                               const zisa::shape_t<3> *to_shapes,
                               int from_rank,
                               int to_rank,
                               zisa::int_t j_offset,
                               cudaStream_t stream = 0);
void transpose_cuda_preprocess(const zisa::array_const_view<complex_t, 3> &from,
                               const zisa::array_view<complex_t, 4> &sendbuf,
                               const zisa::shape_t<3> *from_shapes,
                               const zisa::shape_t<3> *to_shapes,
                               int rank);
void transpose_cuda_preprocess(const zisa::array_const_view<complex_t, 4> &from,
                               const zisa::array_view<complex_t, 5> &sendbuf,
                               const zisa::shape_t<4> *from_shapes,
                               const zisa::shape_t<4> *to_shapes,
                               int from_rank,
                               int to_rank,
                               zisa::int_t k_offset,
                               cudaStream_t stream = 0);
void transpose_cuda_preprocess(const zisa::array_const_view<complex_t, 4> &from,
                               const zisa::array_view<complex_t, 5> &sendbuf,
                               const zisa::shape_t<4> *from_shapes,
                               const zisa::shape_t<4> *to_shapes,
                               int rank);
void transpose_cuda_postprocess(
    const zisa::array_const_view<complex_t, 4> &recvbuf,
    const zisa::array_view<complex_t, 3> &to,
    const zisa::shape_t<3> *from_shapes,
    const zisa::shape_t<3> *to_shapes,
    int from_rank,
    int to_rank,
    zisa::int_t j_offset,
    cudaStream_t stream = 0);
void transpose_cuda_postprocess(
    const zisa::array_const_view<complex_t, 4> &recvbuf,
    const zisa::array_view<complex_t, 3> &to,
    const zisa::shape_t<3> *from_shapes,
    const zisa::shape_t<3> *to_shapes,
    int rank);
void transpose_cuda_postprocess(
    const zisa::array_const_view<complex_t, 5> &recvbuf,
    const zisa::array_view<complex_t, 4> &to,
    const zisa::shape_t<4> *from_shapes,
    const zisa::shape_t<4> *to_shapes,
    int from_rank,
    int to_rank,
    zisa::int_t k_offset,
    cudaStream_t stream = 0);
void transpose_cuda_postprocess(
    const zisa::array_const_view<complex_t, 5> &recvbuf,
    const zisa::array_view<complex_t, 4> &to,
    const zisa::shape_t<4> *from_shapes,
    const zisa::shape_t<4> *to_shapes,
    int rank);

}

#endif
