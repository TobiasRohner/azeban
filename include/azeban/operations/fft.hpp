#ifndef FFT_H_
#define FFT_H_

#include "fft_base.hpp"
#include "fftwfft.hpp"
#ifdef ZISA_HAS_CUDA
#include <azeban/cuda/operations/cufft.hpp>
#endif
#include "fft_factory.hpp"
#include <string>

namespace azeban {

// If no benchmark file exists, leave the filename empty
zisa::int_t optimal_fft_size(const std::string &benchmark_file,
                             zisa::int_t N,
                             int dim,
                             int n_vars,
                             zisa::device_type device);

}

#endif
