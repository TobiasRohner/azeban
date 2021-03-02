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
