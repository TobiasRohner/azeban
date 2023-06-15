#ifndef AZEBAN_UTILS_MATH_
#define AZEBAN_UTILS_MATH_

#include <cmath>

namespace azeban {

ANY_DEVICE_INLINE int abs(int i) {
#ifdef __CUDA_ARCH__
  return :: ::abs(i);
#else
  return std::abs(i);
#endif
}

ANY_DEVICE_INLINE long abs(long i) {
#ifdef __CUDA_ARCH__
  return :: ::labs(i);
#else
  return std::abs(i);
#endif
}

ANY_DEVICE_INLINE long long abs(long long i) {
#ifdef __CUDA_ARCH__
  return :: ::llabs(i);
#else
  return std::abs(i);
#endif
}

ANY_DEVICE_INLINE float abs(float x) {
#ifdef __CUDA_ARCH__
  return :: ::fabsf(x);
#else
  return std::abs(x);
#endif
}

ANY_DEVICE_INLINE double abs(double x) {
#ifdef __CUDA_ARCH__
  return :: ::fabs(x);
#else
  return std::abs(x);
#endif
}

}

#endif
