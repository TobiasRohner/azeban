#ifndef AZEBAN_UTILS_MATH_
#define AZEBAN_UTILS_MATH_

#include <cmath>

namespace azeban {

ANY_DEVICE_INLINE int abs(int i) {
#ifdef __CUDA_ARCH__
  return ::abs(i);
#else
  return std::abs(i);
#endif
}

ANY_DEVICE_INLINE long abs(long i) {
#ifdef __CUDA_ARCH__
  return ::labs(i);
#else
  return std::abs(i);
#endif
}

ANY_DEVICE_INLINE long long abs(long long i) {
#ifdef __CUDA_ARCH__
  return ::llabs(i);
#else
  return std::abs(i);
#endif
}

ANY_DEVICE_INLINE float abs(float x) {
#ifdef __CUDA_ARCH__
  return ::fabsf(x);
#else
  return std::abs(x);
#endif
}

ANY_DEVICE_INLINE double abs(double x) {
#ifdef __CUDA_ARCH__
  return ::fabs(x);
#else
  return std::abs(x);
#endif
}

ANY_DEVICE_INLINE float rhypot(float x, float y) {
#ifdef __CUDA_ARCH__
  return ::rhypotf(x, y);
#else
  return 1.f / std::sqrt(x * x + y * y);
#endif
}

ANY_DEVICE_INLINE double rhypot(double x, double y) {
#ifdef __CUDA_ARCH__
  return ::rhypot(x, y);
#else
  return 1. / std::sqrt(x * x + y * y);
#endif
}

ANY_DEVICE_INLINE float rhypot(float x, float y, float z) {
#ifdef __CUDA_ARCH__
  return ::rsqrtf(x * x + y * y + z * z);
#else
  return 1.f / std::sqrt(x * x + y * y + z * z);
#endif
}

ANY_DEVICE_INLINE double rhypot(double x, double y, double z) {
#ifdef __CUDA_ARCH__
  return ::rsqrt(x * x + y * y + z * z);
#else
  return 1. / std::sqrt(x * x + y * y + z * z);
#endif
}

}

#endif
