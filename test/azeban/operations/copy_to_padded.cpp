#include <azeban/catch.hpp>
#include <azeban/grid.hpp>
#include <azeban/operations/copy_to_padded.hpp>
#include <iostream>
#include <map>
#include <zisa/cuda/memory/cuda_array.hpp>

static long to_real_k(zisa::int_t k, zisa::int_t N) {
  long k_ = zisa::integer_cast<long>(k);
  if (k_ >= zisa::integer_cast<long>(N / 2 + 1)) {
    k_ -= N;
  }
  return k_;
}

static void
test_is_zero_padded(const zisa::array<azeban::complex_t, 1> &unpadded,
                    const zisa::array<azeban::complex_t, 1> &padded) {
  std::map<zisa::int_t, azeban::complex_t> m;
  for (zisa::int_t i = 0; i < unpadded.shape(0); ++i) {
    m[i] = unpadded(i);
  }
  for (zisa::int_t i = 0; i < padded.shape(0); ++i) {
    if (const auto it = m.find(i); it != m.end()) {
      REQUIRE(it->second == padded(i));
    } else {
      REQUIRE(padded(i) == 0);
    }
  }
}

static void
test_is_zero_padded(const zisa::array<azeban::complex_t, 2> &unpadded,
                    const zisa::array<azeban::complex_t, 2> &padded) {
  std::map<std::array<long, 2>, azeban::complex_t> m;
  for (zisa::int_t i = 0; i < unpadded.shape(0); ++i) {
    for (zisa::int_t j = 0; j < unpadded.shape(1); ++j) {
      const std::array<long, 2> kvec{to_real_k(i, unpadded.shape(0)),
                                     to_real_k(j, unpadded.shape(0))};
      m[kvec] = unpadded(i, j);
    }
  }
  for (zisa::int_t i = 0; i < padded.shape(0); ++i) {
    for (zisa::int_t j = 0; j < padded.shape(1); ++j) {
      std::array<long, 2> kvec;
      kvec[0] = to_real_k(i, padded.shape(0));
      kvec[1] = to_real_k(j, padded.shape(0));
      if (const auto it = m.find(kvec); it != m.end()) {
        REQUIRE(it->second == padded(i, j));
      } else {
        REQUIRE(padded(i, j) == 0);
      }
    }
  }
}

static void
test_is_zero_padded(const zisa::array<azeban::complex_t, 3> &unpadded,
                    const zisa::array<azeban::complex_t, 3> &padded) {
  std::map<std::array<long, 3>, azeban::complex_t> m;
  for (zisa::int_t i = 0; i < unpadded.shape(0); ++i) {
    for (zisa::int_t j = 0; j < unpadded.shape(1); ++j) {
      for (zisa::int_t k = 0; k < unpadded.shape(2); ++k) {
        const std::array<long, 3> kvec{to_real_k(i, unpadded.shape(0)),
                                       to_real_k(j, unpadded.shape(0)),
                                       to_real_k(k, unpadded.shape(0))};
        m[kvec] = unpadded(i, j, k);
      }
    }
  }
  for (zisa::int_t i = 0; i < padded.shape(0); ++i) {
    for (zisa::int_t j = 0; j < padded.shape(1); ++j) {
      for (zisa::int_t k = 0; k < padded.shape(2); ++k) {
        const std::array<long, 3> kvec{to_real_k(i, padded.shape(0)),
                                       to_real_k(j, padded.shape(0)),
                                       to_real_k(k, padded.shape(0))};
        if (const auto it = m.find(kvec); it != m.end()) {
          REQUIRE(it->second == padded(i, j, k));
        } else {
          REQUIRE(padded(i, j, k) == 0);
        }
      }
    }
  }
}

template <int Dim>
static void test_zero_padding(zisa::int_t N_unpadded,
                              zisa::int_t N_padded,
                              zisa::device_type device) {
  azeban::Grid<Dim> grid(N_unpadded, N_padded);
  zisa::shape_t<Dim + 1> shape_unpadded = grid.shape_fourier(1);
  zisa::shape_t<Dim + 1> shape_padded = grid.shape_fourier_pad(1);
  zisa::shape_t<Dim> unpadded;
  zisa::shape_t<Dim> padded;
  for (zisa::int_t i = 0; i < Dim; ++i) {
    unpadded[i] = shape_unpadded[i + 1];
    padded[i] = shape_padded[i + 1];
  }
  zisa::array<azeban::complex_t, Dim> h_unpadded_arr(unpadded);
  zisa::array<azeban::complex_t, Dim> h_padded_arr(padded);
  zisa::array<azeban::complex_t, Dim> unpadded_arr(unpadded, device);
  zisa::array<azeban::complex_t, Dim> padded_arr(padded, device);

  std::mt19937 rng;
  std::uniform_real_distribution<azeban::real_t> dist(-1, 1);
  for (azeban::complex_t &c : h_unpadded_arr) {
    c.x = dist(rng);
    c.y = dist(rng);
  }

  zisa::copy(unpadded_arr, h_unpadded_arr);
  azeban::copy_to_padded(padded_arr, unpadded_arr, azeban::complex_t(0));
  zisa::copy(h_padded_arr, padded_arr);

  test_is_zero_padded(h_unpadded_arr, h_padded_arr);
}

TEST_CASE("Zero Padding 1D CPU", "[operations][copy_to_padded]") {
  std::cout << "TESTING: Zero Padding 1D CPU [operations][copy_to_padded]"
            << std::endl;
  for (zisa::int_t N = 8; N <= 1024; N <<= 1) {
    test_zero_padding<1>(N, N, zisa::device_type::cpu);
    test_zero_padding<1>(N, 3 * N / 2, zisa::device_type::cpu);
    test_zero_padding<1>(N, 2 * N, zisa::device_type::cpu);
  }
}

TEST_CASE("Zero Padding 1D GPU", "[operations][copy_to_padded]") {
  std::cout << "TESTING: Zero Padding 1D GPU [operations][copy_to_padded]"
            << std::endl;
  for (zisa::int_t N = 8; N <= 1024; N <<= 1) {
    test_zero_padding<1>(N, N, zisa::device_type::cuda);
    test_zero_padding<1>(N, 3 * N / 2, zisa::device_type::cuda);
    test_zero_padding<1>(N, 2 * N, zisa::device_type::cuda);
  }
}

TEST_CASE("Zero Padding 2D CPU", "[operations][copy_to_padded]") {
  std::cout << "TESTING: Zero Padding 2D CPU [operations][copy_to_padded]"
            << std::endl;
  for (zisa::int_t N = 8; N <= 1024; N <<= 1) {
    test_zero_padding<2>(N, N, zisa::device_type::cpu);
    test_zero_padding<2>(N, 3 * N / 2, zisa::device_type::cpu);
    test_zero_padding<2>(N, 2 * N, zisa::device_type::cpu);
  }
}

TEST_CASE("Zero Padding 2D GPU", "[operations][copy_to_padded]") {
  std::cout << "TESTING: Zero Padding 2D GPU [operations][copy_to_padded]"
            << std::endl;
  for (zisa::int_t N = 8; N <= 1024; N <<= 1) {
    test_zero_padding<2>(N, N, zisa::device_type::cuda);
    test_zero_padding<2>(N, 3 * N / 2, zisa::device_type::cuda);
    test_zero_padding<2>(N, 2 * N, zisa::device_type::cuda);
  }
}

TEST_CASE("Zero Padding 3D CPU", "[operations][copy_to_padded]") {
  std::cout << "TESTING: Zero Padding 3D CPU [operations][copy_to_padded]"
            << std::endl;
  for (zisa::int_t N = 8; N <= 64; N <<= 1) {
    test_zero_padding<3>(N, N, zisa::device_type::cpu);
    test_zero_padding<3>(N, 3 * N / 2, zisa::device_type::cpu);
    test_zero_padding<3>(N, 2 * N, zisa::device_type::cpu);
  }
}

TEST_CASE("Zero Padding 3D GPU", "[operations][copy_to_padded]") {
  std::cout << "TESTING: Zero Padding 3D GPU [operations][copy_to_padded]"
            << std::endl;
  for (zisa::int_t N = 8; N <= 64; N <<= 1) {
    test_zero_padding<3>(N, N, zisa::device_type::cuda);
    test_zero_padding<3>(N, 3 * N / 2, zisa::device_type::cuda);
    test_zero_padding<3>(N, 2 * N, zisa::device_type::cuda);
  }
}
