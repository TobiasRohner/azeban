#include <azeban/cuda/cuda_check_error.hpp>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace azeban::internal {

static const char *cufftGetErrorString(cufftResult err) {
  switch (err) {
  case CUFFT_SUCCESS:
    return "CUFFT_SUCCESS";
  case CUFFT_INVALID_PLAN:
    return "CUFFT_INVALID_PLAN";
  case CUFFT_ALLOC_FAILED:
    return "CUFFT_ALLOC_FAILED";
  case CUFFT_INVALID_TYPE:
    return "CUFFT_INVALID_TYPE";
  case CUFFT_INVALID_VALUE:
    return "CUFFT_INVALID_VALUE";
  case CUFFT_INTERNAL_ERROR:
    return "CUFFT_INTERNAL_ERROR";
  case CUFFT_EXEC_FAILED:
    return "CUFFT_EXEC_FAILED";
  case CUFFT_SETUP_FAILED:
    return "CUFFT_SETUP_FAILED";
  case CUFFT_INVALID_SIZE:
    return "CUFFT_INVALID_SIZE";
  case CUFFT_UNALIGNED_DATA:
    return "CUFFT_UNALIGNED_DATA";
  case CUFFT_INCOMPLETE_PARAMETER_LIST:
    return "CUFFT_INCOMPLETE_PARAMETER_LIST";
  case CUFFT_INVALID_DEVICE:
    return "CUFFT_INVALID_DEVICE";
  case CUFFT_PARSE_ERROR:
    return "CUFFT_PARSE_ERROR";
  case CUFFT_NO_WORKSPACE:
    return "CUFFT_NO_WORKSPACE";
  case CUFFT_NOT_IMPLEMENTED:
    return "CUFFT_NOT_IMPLEMENTED";
  case CUFFT_LICENSE_ERROR:
    return "CUFFT_LICENSE_ERROR";
  case CUFFT_NOT_SUPPORTED:
    return "CUFFT_NOT_SUPPORTED";
  default:
    return "Unknown Error Code";
  }
}

void cudaCheckErrorImpl(cudaError err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr,
            "CUDA Error at %s:%i : %s\n",
            file,
            line,
            cudaGetErrorString(err));
    exit(-1);
  }
}

void cudaCheckErrorImpl(cufftResult err, const char *file, int line) {
  if (err != CUFFT_SUCCESS) {
    fprintf(stderr,
            "CUFFT Error at %s:%i : %s\n",
            file,
            line,
            cufftGetErrorString(err));
    exit(-1);
  }
}

}
