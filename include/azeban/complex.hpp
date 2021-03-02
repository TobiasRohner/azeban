#ifndef COMPLEX_H_
#define COMPLEX_H_

#include <iostream>
#include <zisa/config.hpp>
#include <zisa/math/basic_functions.hpp>

namespace azeban {

template <typename Scalar>
struct Complex {
  using scalar_t = Scalar;

  scalar_t x, y;

  ANY_DEVICE_INLINE Complex() = default;
  ANY_DEVICE_INLINE Complex(scalar_t real, scalar_t imag = 0)
      : x(real), y(imag) {}
  ANY_DEVICE_INLINE Complex(const Complex &other) : x(other.x), y(other.y) {}

  ANY_DEVICE_INLINE Complex &operator=(scalar_t real) {
    x = real;
    y = 0;
    return *this;
  }

  ANY_DEVICE_INLINE Complex &operator=(const Complex &other) {
    x = other.x;
    y = other.y;
    return *this;
  }

  ANY_DEVICE_INLINE Complex &operator==(const Complex &other) {
    return x == other.x && y == other.y;
  }

  ANY_DEVICE_INLINE Complex &operator==(scalar_t real) {
    return x == real && y == 0;
  }

  ANY_DEVICE_INLINE Complex &operator!=(const Complex &other) {
    return x != other.x || y != other.y;
  }

  ANY_DEVICE_INLINE Complex &operator!=(scalar_t real) {
    return x != real || y != 0;
  }

  ANY_DEVICE_INLINE Complex &operator+=(const Complex &other) {
    x += other.x;
    y += other.y;
    return *this;
  }

  ANY_DEVICE_INLINE Complex &operator+=(scalar_t real) {
    x += real;
    return *this;
  }

  ANY_DEVICE_INLINE Complex &operator-=(const Complex &other) {
    x -= other.x;
    y -= other.y;
    return *this;
  }

  ANY_DEVICE_INLINE Complex &operator-=(scalar_t real) {
    x -= real;
    return *this;
  }

  ANY_DEVICE_INLINE Complex &operator*=(const Complex &other) {
    const scalar_t x_ = x;
    x = x * other.x - y * other.y;
    y = x_ * other.y + y * other.x;
    return *this;
  }

  ANY_DEVICE_INLINE Complex &operator*=(scalar_t real) {
    x *= real;
    y *= real;
    return *this;
  }

  ANY_DEVICE_INLINE Complex &operator/=(const Complex &other) {
    const Scalar norm = other.x * other.x + other.y * other.y;
    const scalar_t x_ = x;
    x = (x * other.x + y * other.y) / norm;
    y = (y * other.x - x_ * other.y) / norm;
    return *this;
  }

  ANY_DEVICE_INLINE Complex &operator/=(scalar_t real) {
    x /= real;
    y /= real;
    return *this;
  }
};

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator+(const Complex<Scalar> &c) {
  return c;
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator+(Complex<Scalar> c1,
                                            const Complex<Scalar> &c2) {
  return c1 += c2;
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator+(Complex<Scalar> c, Scalar real) {
  return c += real;
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator+(Scalar real, Complex<Scalar> c) {
  return c += real;
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator-(const Complex<Scalar> &c) {
  return {-c.x, -c.y};
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator-(Complex<Scalar> c1,
                                            const Complex<Scalar> &c2) {
  return c1 -= c2;
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator-(Complex<Scalar> c, Scalar real) {
  return c -= real;
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator-(Scalar real,
                                            const Complex<Scalar> &c) {
  return real + (-c);
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator*(Complex<Scalar> c1,
                                            const Complex<Scalar> &c2) {
  return c1 *= c2;
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator*(Complex<Scalar> c, Scalar real) {
  return c *= real;
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator*(Scalar real, Complex<Scalar> c) {
  return c *= real;
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator/(Complex<Scalar> c1,
                                            const Complex<Scalar> &c2) {
  return c1 /= c2;
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator/(Complex<Scalar> c, Scalar real) {
  return c /= real;
}

template <typename Scalar>
ANY_DEVICE_INLINE Complex<Scalar> operator/(Scalar real, Complex<Scalar> c) {
  return Complex<Scalar>(real) / c;
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

#endif
