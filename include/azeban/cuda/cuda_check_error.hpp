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
#ifndef CUDA_CHECK_ERROR_H_
#define CUDA_CHECK_ERROR_H_

#include <cuda_runtime.h>
#include <cufft.h>

#define cudaCheckError(err)                                                    \
  azeban::internal::cudaCheckErrorImpl(err, __FILE__, __LINE__)

namespace azeban::internal {

void cudaCheckErrorImpl(cudaError err, const char *file, int line);
void cudaCheckErrorImpl(cufftResult err, const char *file, int line);

}

#endif
