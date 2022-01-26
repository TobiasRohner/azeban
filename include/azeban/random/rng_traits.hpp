#ifndef AZEBAN_RANDOM_RNG_TRAITS_HPP_
#define AZEBAN_RANDOM_RNG_TRAITS_HPP_

#include <random>
#include <zisa/memory/device_type.hpp>
#if ZISA_HAS_CUDA
#include <curand.h>
#include <curand_kernel.h>
#endif

namespace azeban {

template <typename RNG>
struct RNGTraits {
  static_assert(!std::is_same<RNG, RNG>::value,
                "Unsupported Random Number Generator");
};

template <typename UIntType, UIntType a, UIntType b, UIntType c>
struct RNGTraits<std::linear_congruential_engine<UIntType, a, b, c>> {
  static constexpr zisa::device_type location = zisa::device_type::cpu;
  using state_t = std::linear_congruential_engine<UIntType, a, b, c>;
};

template <typename UIntType,
          size_t w,
          size_t n,
          size_t m,
          size_t r,
          UIntType a,
          size_t u,
          UIntType d,
          size_t s,
          UIntType b,
          size_t t,
          UIntType c,
          size_t l,
          UIntType f>
struct RNGTraits<std::mersenne_twister_engine<UIntType,
                                              w,
                                              n,
                                              m,
                                              r,
                                              a,
                                              u,
                                              d,
                                              s,
                                              b,
                                              t,
                                              c,
                                              l,
                                              f>> {
  static constexpr zisa::device_type location = zisa::device_type::cpu;
  using state_t = std::
      mersenne_twister_engine<UIntType, w, n, m, r, a, u, d, s, b, t, c, l, f>;
};

template <typename UIntType, size_t w, size_t s, size_t r>
struct RNGTraits<std::subtract_with_carry_engine<UIntType, w, s, r>> {
  static constexpr zisa::device_type location = zisa::device_type::cpu;
  using state_t = std::subtract_with_carry_engine<UIntType, w, s, r>;
};

template <typename Engine, size_t P, size_t R>
struct RNGTraits<std::discard_block_engine<Engine, P, R>> {
  static constexpr zisa::device_type location = zisa::device_type::cpu;
  using state_t = std::discard_block_engine<Engine, P, R>;
};

template <typename Engine, size_t W, typename UIntType>
struct RNGTraits<std::independent_bits_engine<Engine, W, UIntType>> {
  static constexpr zisa::device_type location = zisa::device_type::cpu;
  using state_t = std::independent_bits_engine<Engine, W, UIntType>;
};

template <typename Engine, size_t K>
struct RNGTraits<std::shuffle_order_engine<Engine, K>> {
  static constexpr zisa::device_type location = zisa::device_type::cpu;
  using state_t = std::shuffle_order_engine<Engine, K>;
};

template <>
struct RNGTraits<std::random_device> {
  static constexpr zisa::device_type location = zisa::device_type::cpu;
  using state_t = std::random_device;
};

#if ZISA_HAS_CUDA

template <>
struct RNGTraits<curandStateMtgp32_t> {
  static constexpr zisa::device_type location = zisa::device_type::cuda;
  using state_t = curandStateMtgp32_t;
};

template <>
struct RNGTraits<curandStateScrambledSobol64_t> {
  static constexpr zisa::device_type location = zisa::device_type::cuda;
  using state_t = curandStateScrambledSobol64_t;
};

template <>
struct RNGTraits<curandStateSobol64_t> {
  static constexpr zisa::device_type location = zisa::device_type::cuda;
  using state_t = curandStateSobol64_t;
};

template <>
struct RNGTraits<curandStateScrambledSobol32_t> {
  static constexpr zisa::device_type location = zisa::device_type::cuda;
  using state_t = curandStateScrambledSobol32_t;
};

template <>
struct RNGTraits<curandStateSobol32_t> {
  static constexpr zisa::device_type location = zisa::device_type::cuda;
  using state_t = curandStateSobol32_t;
};

template <>
struct RNGTraits<curandStateMRG32k3a_t> {
  static constexpr zisa::device_type location = zisa::device_type::cuda;
  using state_t = curandStateMRG32k3a_t;
};

template <>
struct RNGTraits<curandStateXORWOW_t> {
  static constexpr zisa::device_type location = zisa::device_type::cuda;
  using state_t = curandStateXORWOW_t;
};

#endif

}

#endif
