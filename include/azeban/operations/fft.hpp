/*
 * This file is part of azeban (https://github.com/TobiasRohner/azeban).
 * Copyright (c) 2021 Tobias Rohner.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef FFT_H_
#define FFT_H_

#include "fft_base.hpp"
#include "fftwfft.hpp"
#if ZISA_HAS_CUDA
#include <azeban/cuda/operations/cufft.hpp>
#endif
#if AZEBAN_HAS_MPI
#include <azeban/cuda/operations/cufft_mpi.hpp>
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
