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
#ifndef COMPLEX_H_
#define COMPLEX_H_

#include <iostream>
#include <zisa/config.hpp>
#include <zisa/math/basic_functions.hpp>
#ifndef __NVCC__
#include <fmt/format.h>
#endif

namespace azeban {

template <typename Scalar>
struct Complex {
  using scalar_t = Scalar;

  scalar_t x, y;

  ANY_DEVICE_INLINE Complex(){};
  ANY_DEVICE_INLINE Complex(scalar_t real, scalar_t imag = 0)
      : x(real), y(imag) {}
  template <typename ScalarR>
  ANY_DEVICE_INLINE Complex(const Complex<ScalarR> &other)
      : x(other.x), y(other.y) {}

  template <typename ScalarR>
  ANY_DEVICE_INLINE Complex &operator=(ScalarR real) {
    x = real;
    y = 0;
    return *this;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE Complex &operator=(const Complex<ScalarR> &other) {
    x = other.x;
    y = other.y;
    return *this;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE bool operator==(const Complex<ScalarR> &other) const {
    return x == other.x && y == other.y;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE bool operator==(ScalarR real) const {
    return x == real && y == 0;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE bool operator!=(const Complex<ScalarR> &other) const {
    return x != other.x || y != other.y;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE bool operator!=(ScalarR real) const {
    return x != real || y != 0;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE Complex &operator+=(const Complex<ScalarR> &other) {
    x += other.x;
    y += other.y;
    return *this;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE Complex &operator+=(ScalarR real) {
    x += real;
    return *this;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE Complex &operator-=(const Complex<ScalarR> &other) {
    x -= other.x;
    y -= other.y;
    return *this;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE Complex &operator-=(ScalarR real) {
    x -= real;
    return *this;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE Complex &operator*=(const Complex<ScalarR> &other) {
    const scalar_t x_ = x;
    x = x * other.x - y * other.y;
    y = x_ * other.y + y * other.x;
    return *this;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE Complex &operator*=(ScalarR real) {
    x *= real;
    y *= real;
    return *this;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE Complex &operator/=(const Complex<ScalarR> &other) {
    const auto norm = other.x * other.x + other.y * other.y;
    const scalar_t x_ = x;
    x = (x * other.x + y * other.y) / norm;
    y = (y * other.x - x_ * other.y) / norm;
    return *this;
  }

  template <typename ScalarR>
  ANY_DEVICE_INLINE Complex &operator/=(ScalarR real) {
    x /= real;
    y /= real;
    return *this;
  }
};

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator+(const Complex<Scalar> &c) {
  return c;
}

template <typename ScalarL, typename ScalarR>
ANY_DEVICE_INLINE auto operator+(const Complex<ScalarL> &c1,
                                 const Complex<ScalarR> &c2)
    -> Complex<decltype(std::declval<ScalarL>() + std::declval<ScalarR>())> {
  using res_t = decltype(std::declval<ScalarL>() + std::declval<ScalarR>());
  Complex<res_t> res = c1;
  return res += c2;
}

template <typename ScalarL, typename ScalarR>
ANY_DEVICE_INLINE auto operator+(const Complex<ScalarL> &c, ScalarR real)
    -> Complex<decltype(std::declval<ScalarL>() + std::declval<ScalarR>())> {
  using res_t = decltype(std::declval<ScalarL>() + std::declval<ScalarR>());
  Complex<res_t> res = c;
  return res += real;
}

template <typename ScalarL, typename ScalarR>
ANY_DEVICE_INLINE auto operator+(ScalarL real, const Complex<ScalarR> &c)
    -> Complex<decltype(std::declval<ScalarL>() + std::declval<ScalarR>())> {
  using res_t = decltype(std::declval<ScalarL>() + std::declval<ScalarR>());
  Complex<res_t> res = c;
  return res += real;
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator-(const Complex<Scalar> &c) {
  return {-c.x, -c.y};
}

template <typename ScalarL, typename ScalarR>
ANY_DEVICE_INLINE auto operator-(const Complex<ScalarL> &c1,
                                 const Complex<ScalarR> &c2)
    -> Complex<decltype(std::declval<ScalarL>() - std::declval<ScalarR>())> {
  using res_t = decltype(std::declval<ScalarL>() - std::declval<ScalarR>());
  Complex<res_t> res = c1;
  return res -= c2;
}

template <typename ScalarL, typename ScalarR>
ANY_DEVICE_INLINE auto operator-(const Complex<ScalarL> &c, ScalarR real)
    -> Complex<decltype(std::declval<ScalarL>() - std::declval<ScalarR>())> {
  using res_t = decltype(std::declval<ScalarL>() - std::declval<ScalarR>());
  Complex<res_t> res = c;
  return res -= real;
}

template <typename ScalarL, typename ScalarR>
ANY_DEVICE_INLINE auto operator-(ScalarL real, const Complex<ScalarR> &c)
    -> Complex<decltype(std::declval<ScalarL>() - std::declval<ScalarR>())> {
  using res_t = decltype(std::declval<ScalarL>() - std::declval<ScalarR>());
  Complex<res_t> res = real;
  return res -= c;
}

template <typename ScalarL, typename ScalarR>
ANY_DEVICE_INLINE auto operator*(const Complex<ScalarL> &c1,
                                 const Complex<ScalarR> &c2)
    -> Complex<decltype(std::declval<ScalarL>() * std::declval<ScalarR>())> {
  using res_t = decltype(std::declval<ScalarL>() * std::declval<ScalarR>());
  Complex<res_t> res = c1;
  return res *= c2;
}

template <typename ScalarL, typename ScalarR>
ANY_DEVICE_INLINE auto operator*(const Complex<ScalarL> &c, ScalarR real)
    -> Complex<decltype(std::declval<ScalarL>() * std::declval<ScalarR>())> {
  using res_t = decltype(std::declval<ScalarL>() * std::declval<ScalarR>());
  Complex<res_t> res = c;
  return res *= real;
}

template <typename ScalarL, typename ScalarR>
ANY_DEVICE_INLINE auto operator*(ScalarL real, const Complex<ScalarR> &c)
    -> Complex<decltype(std::declval<ScalarL>() * std::declval<ScalarR>())> {
  using res_t = decltype(std::declval<ScalarL>() * std::declval<ScalarR>());
  Complex<res_t> res = c;
  return res *= real;
}

template <typename ScalarL, typename ScalarR>
ANY_DEVICE_INLINE auto operator/(const Complex<ScalarL> &c1,
                                 const Complex<ScalarR> &c2)
    -> Complex<decltype(std::declval<ScalarL>() / std::declval<ScalarR>())> {
  using res_t = decltype(std::declval<ScalarL>() / std::declval<ScalarR>());
  Complex<res_t> res = c1;
  return res /= c2;
}

template <typename ScalarL, typename ScalarR>
ANY_DEVICE_INLINE auto operator/(const Complex<ScalarL> &c, ScalarR real)
    -> Complex<decltype(std::declval<ScalarL>() / std::declval<ScalarR>())> {
  using res_t = decltype(std::declval<ScalarL>() / std::declval<ScalarR>());
  Complex<res_t> res = c;
  return res /= real;
}

template <typename ScalarL, typename ScalarR>
ANY_DEVICE_INLINE auto operator/(ScalarL real, const Complex<ScalarR> &c)
    -> Complex<decltype(std::declval<ScalarL>() / std::declval<ScalarR>())> {
  using res_t = decltype(std::declval<ScalarL>() / std::declval<ScalarR>());
  Complex<res_t> res = real;
  return res /= c;
}

template <typename Scalar>
ANY_DEVICE_INLINE Scalar abs2(const Complex<Scalar> &c) {
  return c.x * c.x + c.y * c.y;
}

template <typename Scalar>
ANY_DEVICE_INLINE Scalar abs(const Complex<Scalar> &c) {
  return zisa::sqrt(abs2(c));
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &os, const Complex<Scalar> &c) {
  return os << '(' << c.x << ", " << c.y << ')';
}

}

#ifndef __NVCC__
template <typename Scalar>
struct fmt::formatter<azeban::Complex<Scalar>> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const azeban::Complex<Scalar> &c, FormatContext &ctx) {
    return fmt::format_to(ctx.out(), "({}, {})", c.x, c.y);
  }
};
#endif

#endif
